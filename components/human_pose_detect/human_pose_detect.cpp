#include "human_pose_detect.hpp"

#include <algorithm>
#include <math.h>
#include <string>

#include "dl_define.hpp"
#include "dl_image_preprocessor.hpp"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "fbs_model.hpp"

extern const uint8_t movenet_lightning_192_int8_espdl_start[] asm("_binary_movenet_lightning_192_int8_espdl_start");
extern const uint8_t movenet_lightning_192_int8_espdl_end[] asm("_binary_movenet_lightning_192_int8_espdl_end");

static const char *TAG = "pose_detect";
static constexpr int MOVENET_KEYPOINTS = 17;
static constexpr int OVERLAY_KEYPOINTS = 33;
static constexpr int RAW_HEAD_SIZE = 48;
static constexpr float MOVENET_VISIBLE_SCORE_THR = 0.18f;
static constexpr int MOVENET_KEYPOINT_SEARCH_RADIUS = 8;
static constexpr bool MOVENET_PROFILE_FIRST_FRAME = false;
static constexpr size_t MOVENET_INTERNAL_ARENA_RESERVE = 32 * 1024;
static constexpr size_t MOVENET_INTERNAL_ARENA_MAX = 192 * 1024;
static constexpr dl::runtime_mode_t MOVENET_RUNTIME_MODE = dl::RUNTIME_MODE_AUTO;
static constexpr int MP_LEFT_SHOULDER = 11;
static constexpr int MP_RIGHT_SHOULDER = 12;
static constexpr int MP_LEFT_HIP = 23;
static constexpr int MP_RIGHT_HIP = 24;

typedef struct {
    int mp_idx;
    int coco_idx;
} kpt_map_t;

static constexpr kpt_map_t COCO_TO_MP[] = {
    {0, 0},   {2, 1},   {5, 2},   {7, 3},   {8, 4},   {11, 5},
    {12, 6}, {13, 7},  {14, 8},  {15, 9},  {16, 10}, {23, 11},
    {24, 12}, {25, 13}, {26, 14}, {27, 15}, {28, 16},
};

static std::string shape_to_string(const std::vector<int> &shape)
{
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        out += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            out += ",";
        }
    }
    out += "]";
    return out;
}

static float tensor_scale(dl::TensorBase *tensor)
{
    return tensor ? DL_SCALE((int)tensor->exponent) : 1.0f;
}

static float tensor_value(dl::TensorBase *tensor, int index)
{
    if (!tensor || !tensor->data || index < 0 || index >= tensor->get_size()) {
        return 0.0f;
    }
    const float scale = tensor_scale(tensor);
    switch (tensor->dtype) {
    case dl::DATA_TYPE_INT8:
        return (float)tensor->get_element_ptr<int8_t>()[index] * scale;
    case dl::DATA_TYPE_INT16:
        return (float)tensor->get_element_ptr<int16_t>()[index] * scale;
    case dl::DATA_TYPE_INT32:
        return (float)tensor->get_element_ptr<int32_t>()[index] * scale;
    case dl::DATA_TYPE_FLOAT:
        return tensor->get_element_ptr<float>()[index];
    default:
        return 0.0f;
    }
}

static int choose_internal_arena_limit(size_t largest_internal)
{
    if (largest_internal <= MOVENET_INTERNAL_ARENA_RESERVE) {
        return 0;
    }

    const size_t usable = std::min(largest_internal - MOVENET_INTERNAL_ARENA_RESERVE, MOVENET_INTERNAL_ARENA_MAX);
    if (usable < 16 * 1024) {
        return 0;
    }
    return (int)(usable & ~(size_t)0x3ff);
}

static float sigmoidf_fast(float x)
{
    if (x > 16.0f) {
        return 1.0f;
    }
    if (x < -16.0f) {
        return 0.0f;
    }
    return 1.0f / (1.0f + expf(-x));
}

static float pose_score_at(const pose_result_t &pose, int idx)
{
    if (idx < 0 || idx >= (int)pose.keypoints.size()) {
        return 0.0f;
    }
    return pose.keypoints[idx].score;
}

static float pose_quality_pc_like(const pose_result_t &pose)
{
    if (pose.keypoints.empty()) {
        return -1.0f;
    }

    int valid = 0;
    float score_sum = 0.0f;
    int scored = 0;
    int min_x = INT32_MAX;
    int min_y = INT32_MAX;
    int max_x = INT32_MIN;
    int max_y = INT32_MIN;
    for (const auto &kp : pose.keypoints) {
        score_sum += kp.score;
        scored++;
        if (kp.score >= MOVENET_VISIBLE_SCORE_THR) {
            valid++;
            min_x = std::min(min_x, kp.x);
            min_y = std::min(min_y, kp.y);
            max_x = std::max(max_x, kp.x);
            max_y = std::max(max_y, kp.y);
        }
    }

    const float mean_score = scored > 0 ? score_sum / (float)scored : 0.0f;
    float box_bonus = 0.0f;
    if (valid > 0 && max_x > min_x && max_y > min_y) {
        const float box_w = (float)(max_x - min_x);
        const float box_h = (float)(max_y - min_y);
        const float aspect = box_w / std::max(box_h, 1.0f);
        box_bonus = (aspect >= 0.2f && aspect <= 5.0f) ? 1.0f : 0.0f;
    }
    const float torso_bonus =
        (pose_score_at(pose, MP_LEFT_SHOULDER) >= MOVENET_VISIBLE_SCORE_THR &&
         pose_score_at(pose, MP_RIGHT_SHOULDER) >= MOVENET_VISIBLE_SCORE_THR &&
         pose_score_at(pose, MP_LEFT_HIP) >= MOVENET_VISIBLE_SCORE_THR &&
         pose_score_at(pose, MP_RIGHT_HIP) >= MOVENET_VISIBLE_SCORE_THR)
            ? 1.0f
            : 0.0f;
    return (float)valid * 1000.0f + mean_score * 100.0f + box_bonus * 10.0f + torso_bonus;
}

static inline void unpack_rgb565(dl::image::pix_type_t pix_type, uint16_t pixel, uint8_t &r, uint8_t &g, uint8_t &b)
{
    uint8_t c0 = (pixel >> 11) & 0x1f;
    uint8_t c1 = (pixel >> 5) & 0x3f;
    uint8_t c2 = pixel & 0x1f;
    if (pix_type == dl::image::DL_IMAGE_PIX_TYPE_BGR565LE || pix_type == dl::image::DL_IMAGE_PIX_TYPE_BGR565BE) {
        b = (c0 << 3) | (c0 >> 2);
        g = (c1 << 2) | (c1 >> 4);
        r = (c2 << 3) | (c2 >> 2);
    } else {
        r = (c0 << 3) | (c0 >> 2);
        g = (c1 << 2) | (c1 >> 4);
        b = (c2 << 3) | (c2 >> 2);
    }
}

static bool is_rgb565_type(dl::image::pix_type_t pix_type)
{
    return pix_type == dl::image::DL_IMAGE_PIX_TYPE_RGB565LE || pix_type == dl::image::DL_IMAGE_PIX_TYPE_RGB565BE ||
           pix_type == dl::image::DL_IMAGE_PIX_TYPE_BGR565LE || pix_type == dl::image::DL_IMAGE_PIX_TYPE_BGR565BE;
}

static void preprocess_rgb565_to_raw_float_nhwc(dl::TensorBase *input,
                                                const dl::image::img_t &img,
                                                float &inv_scale_x,
                                                float &inv_scale_y,
                                                int &border_left,
                                                int &border_top,
                                                int &crop_left,
                                                int &crop_top)
{
    const int dst_h = input->shape[1];
    const int dst_w = input->shape[2];
    float *dst = input->get_element_ptr<float>();
    const uint8_t *src = static_cast<const uint8_t *>(img.data);

    const float scale_x = (float)dst_w / (float)img.width;
    const float scale_y = (float)dst_h / (float)img.height;
    const float scale = std::min(scale_x, scale_y);
    const int scaled_w = std::max(1, (int)lroundf((float)img.width * scale));
    const int scaled_h = std::max(1, (int)lroundf((float)img.height * scale));
    border_left = (dst_w - scaled_w) / 2;
    border_top = (dst_h - scaled_h) / 2;
    inv_scale_x = 1.0f / scale;
    inv_scale_y = 1.0f / scale;
    crop_left = 0;
    crop_top = 0;

    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const int out_index = (y * dst_w + x) * 3;
            if (x < border_left || x >= border_left + scaled_w || y < border_top || y >= border_top + scaled_h) {
                dst[out_index + 0] = 0.0f;
                dst[out_index + 1] = 0.0f;
                dst[out_index + 2] = 0.0f;
                continue;
            }

            const float src_fx = ((float)(x - border_left) + 0.5f) * inv_scale_x - 0.5f;
            const float src_fy = ((float)(y - border_top) + 0.5f) * inv_scale_y - 0.5f;
            const int src_x = std::max(0, std::min((int)img.width - 1, (int)lroundf(src_fx)));
            const int src_y = std::max(0, std::min((int)img.height - 1, (int)lroundf(src_fy)));
            const size_t src_index = ((size_t)src_y * img.width + src_x) * 2;
            uint16_t pixel = 0;
            if (img.pix_type == dl::image::DL_IMAGE_PIX_TYPE_RGB565LE ||
                img.pix_type == dl::image::DL_IMAGE_PIX_TYPE_BGR565LE) {
                pixel = (uint16_t)src[src_index] | ((uint16_t)src[src_index + 1] << 8);
            } else {
                pixel = ((uint16_t)src[src_index] << 8) | (uint16_t)src[src_index + 1];
            }

            uint8_t r = 0, g = 0, b = 0;
            unpack_rgb565(img.pix_type, pixel, r, g, b);
            dst[out_index + 0] = (float)r;
            dst[out_index + 1] = (float)g;
            dst[out_index + 2] = (float)b;
        }
    }
}

static int tensor_channels(dl::TensorBase *tensor)
{
    if (!tensor || tensor->shape.size() != 4) {
        return 0;
    }
    if (tensor->shape[3] == 1 || tensor->shape[3] == 17 || tensor->shape[3] == 34) {
        return tensor->shape[3];
    }
    if (tensor->shape[1] == 1 || tensor->shape[1] == 17 || tensor->shape[1] == 34) {
        return tensor->shape[1];
    }
    return 0;
}

static bool tensor_is_nhwc(dl::TensorBase *tensor)
{
    return tensor && tensor->shape.size() == 4 &&
           (tensor->shape[3] == 1 || tensor->shape[3] == 17 || tensor->shape[3] == 34);
}

enum class tensor_data_layout_t {
    NHWC,
    CHW_PACKED,
};

static const char *tensor_layout_name(tensor_data_layout_t layout)
{
    return layout == tensor_data_layout_t::NHWC ? "NHWC" : "CHW_PACKED";
}

static int tensor_channels_for_layout(dl::TensorBase *tensor, tensor_data_layout_t layout)
{
    if (!tensor || tensor->shape.size() != 4) {
        return 0;
    }
    if (layout == tensor_data_layout_t::NHWC && (tensor->shape[3] == 1 || tensor->shape[3] == 17 || tensor->shape[3] == 34)) {
        return tensor->shape[3];
    }
    if (tensor->shape[1] == 1 || tensor->shape[1] == 17 || tensor->shape[1] == 34) {
        return tensor->shape[1];
    }
    if (tensor->shape[3] == 1 || tensor->shape[3] == 17 || tensor->shape[3] == 34) {
        return tensor->shape[3];
    }
    return 0;
}

static int tensor_h_for_layout(dl::TensorBase *tensor, tensor_data_layout_t layout)
{
    if (!tensor || tensor->shape.size() != 4) {
        return 0;
    }
    if (layout == tensor_data_layout_t::NHWC && (tensor->shape[3] == 1 || tensor->shape[3] == 17 || tensor->shape[3] == 34)) {
        return tensor->shape[1];
    }
    if (tensor->shape[1] == 1 || tensor->shape[1] == 17 || tensor->shape[1] == 34) {
        return tensor->shape[2];
    }
    if (tensor->shape[3] == 1 || tensor->shape[3] == 17 || tensor->shape[3] == 34) {
        return tensor->shape[1];
    }
    return 0;
}

static int tensor_w_for_layout(dl::TensorBase *tensor, tensor_data_layout_t layout)
{
    if (!tensor || tensor->shape.size() != 4) {
        return 0;
    }
    if (layout == tensor_data_layout_t::NHWC && (tensor->shape[3] == 1 || tensor->shape[3] == 17 || tensor->shape[3] == 34)) {
        return tensor->shape[2];
    }
    if (tensor->shape[1] == 1 || tensor->shape[1] == 17 || tensor->shape[1] == 34) {
        return tensor->shape[3];
    }
    if (tensor->shape[3] == 1 || tensor->shape[3] == 17 || tensor->shape[3] == 34) {
        return tensor->shape[2];
    }
    return 0;
}

static float tensor_value_4d_for_layout(dl::TensorBase *tensor, int y, int x, int c, tensor_data_layout_t layout)
{
    if (!tensor || !tensor->data || tensor->shape.size() != 4 || tensor->axis_offset.size() != 4) {
        return 0.0f;
    }
    const int h = tensor_h_for_layout(tensor, layout);
    const int w = tensor_w_for_layout(tensor, layout);
    const int ch = tensor_channels_for_layout(tensor, layout);
    if (y < 0 || y >= h || x < 0 || x >= w || c < 0 || c >= ch) {
        return 0.0f;
    }
    if (layout == tensor_data_layout_t::NHWC && tensor_is_nhwc(tensor)) {
        const int index = y * tensor->axis_offset[1] + x * tensor->axis_offset[2] + c * tensor->axis_offset[3];
        return tensor_value(tensor, index);
    }
    const int index = c * tensor->axis_offset[1] + y * tensor->axis_offset[2] + x * tensor->axis_offset[3];
    return tensor_value(tensor, index);
}

HumanPoseDetect::HumanPoseDetect(float score_thr, float nms_thr, int top_k) :
    m_score_thr(score_thr), m_nms_thr(nms_thr), m_top_k(top_k)
{
    (void)m_nms_thr;
    ESP_LOGI(TAG,
             "heap before pose: psram=%u internal=%u largest_psram=%u largest_internal=%u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    ESP_LOGI(TAG,
             "MoveNet model variant: lightning_192 FLOAT/INT8 input + INT8 heads, PC-aligned preprocess + C++ decode + safe runtime");
    ESP_LOGI(TAG,
             "MoveNet embedded model bytes=%u expected_sha256=95FEFE27CB00AA95...",
             (unsigned)(movenet_lightning_192_int8_espdl_end - movenet_lightning_192_int8_espdl_start));

    const size_t largest_internal = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
    const int internal_limit = choose_internal_arena_limit(largest_internal);
    const bool param_copy = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) > (4 * 1024 * 1024);
    m_model = new dl::Model((const char *)movenet_lightning_192_int8_espdl_start,
                            fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                            internal_limit,
                            dl::MEMORY_MANAGER_GREEDY,
                            nullptr,
                            param_copy);
    if (!m_model) {
        m_error = "MoveNet model allocation failed";
        ESP_LOGE(TAG, "%s", m_error);
        return;
    }
    m_model->minimize();

    dl::TensorBase *input = m_model->get_input();
    if (!input || input->shape.size() != 4 || input->shape[1] != 192 || input->shape[2] != 192 || input->shape[3] != 3 ||
        (input->dtype != dl::DATA_TYPE_INT8 && input->dtype != dl::DATA_TYPE_INT16 && input->dtype != dl::DATA_TYPE_FLOAT)) {
        m_error = "MoveNet model incompatible: expected NHWC 1x192x192x3 input with int8/int16/float dtype";
        ESP_LOGE(TAG, "%s", m_error);
        if (input) {
            ESP_LOGE(TAG,
                     "input shape=%s dtype=%s exp=%d",
                     shape_to_string(input->shape).c_str(),
                     input->get_dtype_string(),
                     (int)input->exponent);
        }
        return;
    }

    if (input->dtype == dl::DATA_TYPE_INT8 || input->dtype == dl::DATA_TYPE_INT16) {
        m_preprocessor = new dl::image::ImagePreprocessor(m_model, {127.5f, 127.5f, 127.5f}, {127.5f, 127.5f, 127.5f});
        if (!m_preprocessor) {
            m_error = "MoveNet preprocessor allocation failed";
            ESP_LOGE(TAG, "%s", m_error);
            return;
        }
        m_preprocessor->enable_letterbox({0, 0, 0});
    }

    ESP_LOGI(TAG,
             "input shape=%s dtype=%s exp=%d internal_limit=%d param_copy=%d",
             shape_to_string(input->shape).c_str(),
             input->get_dtype_string(),
             (int)input->exponent,
             internal_limit,
             param_copy ? 1 : 0);
    ESP_LOGI(TAG,
             "runtime=%s profile_first_frame=%d internal_reserve=%u",
             MOVENET_RUNTIME_MODE == dl::RUNTIME_MODE_SINGLE_CORE ? "single-core-safe" : "multi/auto",
             MOVENET_PROFILE_FIRST_FRAME ? 1 : 0,
             (unsigned)MOVENET_INTERNAL_ARENA_RESERVE);
    if (input->dtype == dl::DATA_TYPE_FLOAT) {
        ESP_LOGI(TAG,
                 "using FLOAT input model; board writes raw RGB 0..255 and model applies built-in MoveNet normalization");
    } else {
        ESP_LOGI(TAG,
                 "using quantized input model; ESP-DL ImagePreprocessor applies MoveNet normalization [-1, 1]");
    }
    for (const auto &item : m_model->get_outputs()) {
        dl::TensorBase *out = item.second;
        ESP_LOGI(TAG,
                 "output %s shape=%s axis=%s dtype=%s exp=%d",
                 item.first.c_str(),
                 shape_to_string(out->shape).c_str(),
                 shape_to_string(out->axis_offset).c_str(),
                 out->get_dtype_string(),
                 (int)out->exponent);
    }

    m_ready = true;
    m_error = "";
}

HumanPoseDetect::~HumanPoseDetect()
{
    delete m_preprocessor;
    m_preprocessor = nullptr;
    delete m_model;
    m_model = nullptr;
}

const std::vector<pose_result_t> &HumanPoseDetect::run(const dl::image::img_t &img)
{
    m_results.clear();
    m_last_quality = 0.0f;
    m_last_avg_score = 0.0f;
    m_last_valid_count = 0;
    m_last_preprocess_ms = 0.0f;
    m_last_forward_ms = 0.0f;
    m_last_postprocess_ms = 0.0f;
    if (!m_ready || !m_model || !img.data || img.width == 0 || img.height == 0) {
        return m_results;
    }

    const int64_t pre_start = esp_timer_get_time();
    dl::TensorBase *input = m_model->get_input();
    if (!input) {
        return m_results;
    }
    if (input->dtype == dl::DATA_TYPE_INT8 || input->dtype == dl::DATA_TYPE_INT16) {
        if (!m_preprocessor) {
            return m_results;
        }
        m_preprocessor->preprocess(img);
        m_inv_scale_x = m_preprocessor->get_resize_scale_x(true);
        m_inv_scale_y = m_preprocessor->get_resize_scale_y(true);
        m_border_left = m_preprocessor->get_border_left();
        m_border_top = m_preprocessor->get_border_top();
        m_crop_left = m_preprocessor->get_crop_area_top_left_x();
        m_crop_top = m_preprocessor->get_crop_area_top_left_y();
    } else if (input->dtype == dl::DATA_TYPE_FLOAT) {
        if (!is_rgb565_type(img.pix_type)) {
            ESP_LOGE(TAG, "MoveNet float preprocess unsupported pix_type=%s", dl::image::pix_type2str(img.pix_type).c_str());
            return m_results;
        }
        preprocess_rgb565_to_raw_float_nhwc(
            input, img, m_inv_scale_x, m_inv_scale_y, m_border_left, m_border_top, m_crop_left, m_crop_top);
    } else {
        return m_results;
    }
    const int64_t pre_end = esp_timer_get_time();

    if (MOVENET_PROFILE_FIRST_FRAME && m_profile_pending) {
        ESP_LOGI(TAG, "MoveNet one-shot ESP-DL layer profile begin (first local frame only)");
        m_model->profile(true);
        ESP_LOGI(TAG, "MoveNet one-shot ESP-DL layer profile end");
        m_profile_pending = false;
    }

    const int64_t forward_start = esp_timer_get_time();
    // Keep MoveNet on the stable path. AUTO/dual-core hits the ESP32-P4 s8 unaligned conv
    // split path for this graph and corrupts heap/TLSF on current ESP-DL.
    m_model->run(MOVENET_RUNTIME_MODE);
    const int64_t forward_end = esp_timer_get_time();

    const int64_t post_start = esp_timer_get_time();
    dl::TensorBase *center = nullptr;
    dl::TensorBase *heatmap = nullptr;
    dl::TensorBase *regress = nullptr;
    dl::TensorBase *offset = nullptr;

    for (const auto &item : m_model->get_outputs()) {
        dl::TensorBase *tensor = item.second;
        const int channels = tensor_channels(tensor);
        const std::string &name = item.first;
        if (channels == 1 || name.find("center") != std::string::npos) {
            center = tensor;
        } else if (channels == 17 || name.find("heatmap") != std::string::npos) {
            heatmap = tensor;
        } else if (name.find("regress") != std::string::npos) {
            regress = tensor;
        } else if (name.find("offset") != std::string::npos) {
            offset = tensor;
        }
    }
    for (const auto &item : m_model->get_outputs()) {
        dl::TensorBase *tensor = item.second;
        if (tensor_channels(tensor) == 34) {
            if (!regress) {
                regress = tensor;
            } else if (!offset && tensor != regress) {
                offset = tensor;
            }
        }
    }

    if (!center || !heatmap || !regress || !offset) {
        ESP_LOGE(TAG, "MoveNet output heads missing center=%p heat=%p reg=%p off=%p", center, heatmap, regress, offset);
        return m_results;
    }

    auto decode_layout = [&](tensor_data_layout_t layout,
                             std::vector<pose_result_t> &decoded,
                             int &valid,
                             float &max_score,
                             float &avg_score) -> float {
        decoded.clear();
        valid = 0;
        max_score = 0.0f;
        avg_score = 0.0f;

        const int out_h = tensor_h_for_layout(center, layout);
        const int out_w = tensor_w_for_layout(center, layout);
        if (out_h <= 0 || out_w <= 0 || out_h != tensor_h_for_layout(heatmap, layout) ||
            out_w != tensor_w_for_layout(heatmap, layout)) {
            return -1.0f;
        }

        int center_y = 0;
        int center_x = 0;
        float center_best = -1.0f;
        const float center_ref_y = (float)out_h / 2.0f;
        const float center_ref_x = (float)out_w / 2.0f;
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                const float score = sigmoidf_fast(tensor_value_4d_for_layout(center, y, x, 0, layout));
                const float center_prior = sqrtf(((float)y - center_ref_y) * ((float)y - center_ref_y) +
                                                 ((float)x - center_ref_x) * ((float)x - center_ref_x));
                const float weighted_center = score / (center_prior + 1.8f);
                if (weighted_center > center_best) {
                    center_best = weighted_center;
                    center_y = y;
                    center_x = x;
                }
            }
        }

        pose_keypoint_t coco[MOVENET_KEYPOINTS] = {};
        float score_sum = 0.0f;
        int min_x = (int)img.width - 1;
        int min_y = (int)img.height - 1;
        int max_x = 0;
        int max_y = 0;

        for (int k = 0; k < MOVENET_KEYPOINTS; ++k) {
            const float ry = (float)center_y + tensor_value_4d_for_layout(regress, center_y, center_x, k * 2, layout);
            const float rx = (float)center_x + tensor_value_4d_for_layout(regress, center_y, center_x, k * 2 + 1, layout);
            int best_y = 0;
            int best_x = 0;
            float best_weighted = -1.0f;
            float best_score = 0.0f;

            int search_cy = (int)lroundf(ry);
            int search_cx = (int)lroundf(rx);
            if (!isfinite(ry) || !isfinite(rx)) {
                search_cy = center_y;
                search_cx = center_x;
            }
            search_cy = std::max(0, std::min(out_h - 1, search_cy));
            search_cx = std::max(0, std::min(out_w - 1, search_cx));
            const int y0 = std::max(0, search_cy - MOVENET_KEYPOINT_SEARCH_RADIUS);
            const int y1 = std::min(out_h - 1, search_cy + MOVENET_KEYPOINT_SEARCH_RADIUS);
            const int x0 = std::max(0, search_cx - MOVENET_KEYPOINT_SEARCH_RADIUS);
            const int x1 = std::min(out_w - 1, search_cx + MOVENET_KEYPOINT_SEARCH_RADIUS);

            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    const float heat = sigmoidf_fast(tensor_value_4d_for_layout(heatmap, y, x, k, layout));
                    const float dy = (float)y - ry;
                    const float dx = (float)x - rx;
                    const float weighted = heat / (sqrtf(dx * dx + dy * dy) + 1.8f);
                    if (weighted > best_weighted) {
                        best_weighted = weighted;
                        best_score = heat;
                        best_y = y;
                        best_x = x;
                    }
                }
            }

            const float py = ((float)best_y + tensor_value_4d_for_layout(offset, best_y, best_x, k * 2, layout)) /
                             (float)out_h;
            const float px = ((float)best_x + tensor_value_4d_for_layout(offset, best_y, best_x, k * 2 + 1, layout)) /
                             (float)out_w;
            const float padded_x = px * (float)input->shape[2];
            const float padded_y = py * (float)input->shape[1];
            const int src_x = (int)lroundf(((padded_x - (float)m_border_left) * m_inv_scale_x) + (float)m_crop_left);
            const int src_y = (int)lroundf(((padded_y - (float)m_border_top) * m_inv_scale_y) + (float)m_crop_top);

            coco[k].x = std::max(0, std::min((int)img.width - 1, src_x));
            coco[k].y = std::max(0, std::min((int)img.height - 1, src_y));
            coco[k].score = best_score;
            score_sum += best_score;
            max_score = std::max(max_score, best_score);
            if (best_score >= MOVENET_VISIBLE_SCORE_THR) {
                min_x = std::min(min_x, coco[k].x);
                min_y = std::min(min_y, coco[k].y);
                max_x = std::max(max_x, coco[k].x);
                max_y = std::max(max_y, coco[k].y);
                valid++;
            }
        }

        avg_score = score_sum / (float)MOVENET_KEYPOINTS;
        pose_result_t pose = {};
        pose.score = avg_score;
        if (valid > 0 && max_x > min_x && max_y > min_y) {
            pose.x1 = min_x;
            pose.y1 = min_y;
            pose.x2 = max_x;
            pose.y2 = max_y;
        } else {
            pose.x1 = 0;
            pose.y1 = 0;
            pose.x2 = (int)img.width - 1;
            pose.y2 = (int)img.height - 1;
        }
        pose.keypoints.assign(OVERLAY_KEYPOINTS, {});
        for (const auto &map : COCO_TO_MP) {
            pose.keypoints[map.mp_idx] = coco[map.coco_idx];
        }

        const float pose_quality = pose_quality_pc_like(pose);
        if (max_score >= m_score_thr && valid >= 1) {
            decoded.push_back(pose);
        }
        return pose_quality;
    };

    const tensor_data_layout_t used_layout =
        (tensor_is_nhwc(center) && tensor_is_nhwc(heatmap) && tensor_is_nhwc(regress) && tensor_is_nhwc(offset))
            ? tensor_data_layout_t::NHWC
            : tensor_data_layout_t::CHW_PACKED;
    int valid = 0;
    float max_score = 0.0f;
    float avg_score = 0.0f;
    const float used_quality = decode_layout(used_layout, m_results, valid, max_score, avg_score);

    m_last_valid_count = valid;
    m_last_avg_score = avg_score;
    m_last_quality = used_quality;

    const int64_t post_end = esp_timer_get_time();
    m_last_preprocess_ms = (pre_end - pre_start) / 1000.0f;
    m_last_forward_ms = (forward_end - forward_start) / 1000.0f;
    m_last_postprocess_ms = (post_end - post_start) / 1000.0f;

    static int debug_frame = 0;
    debug_frame++;
    if (debug_frame <= 5 || debug_frame % 10 == 0 || m_results.empty()) {
        ESP_LOGI(TAG,
                 "movenet: pre=%.1f ms fwd=%.1f ms post=%.1f ms valid=%d max=%.3f layout=%s q=%.1f results=%d",
                 (double)m_last_preprocess_ms,
                 (double)m_last_forward_ms,
                 (double)m_last_postprocess_ms,
                 valid,
                 (double)max_score,
                 tensor_layout_name(used_layout),
                 (double)used_quality,
                 (int)m_results.size());
        if (!m_results.empty() && m_results[0].keypoints.size() > 24) {
            const auto &kp = m_results[0].keypoints;
            ESP_LOGI(TAG,
                     "movenet kpts: nose=(%d,%d,%.2f) ls=(%d,%d,%.2f) rs=(%d,%d,%.2f) lh=(%d,%d,%.2f) rh=(%d,%d,%.2f)",
                     kp[0].x,
                     kp[0].y,
                     (double)kp[0].score,
                     kp[11].x,
                     kp[11].y,
                     (double)kp[11].score,
                     kp[12].x,
                     kp[12].y,
                     (double)kp[12].score,
                     kp[23].x,
                     kp[23].y,
                     (double)kp[23].score,
                     kp[24].x,
                     kp[24].y,
                     (double)kp[24].score);
        }
    }

    return m_results;
}
