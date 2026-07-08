#pragma once

#include <stdint.h>
#include <vector>

#include "dl_image_define.hpp"

namespace dl {
class Model;
namespace image {
class ImagePreprocessor;
}
} // namespace dl

typedef struct {
    int x;
    int y;
    float score;
} pose_keypoint_t;

typedef struct {
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    std::vector<pose_keypoint_t> keypoints;
} pose_result_t;

class HumanPoseDetect {
public:
    HumanPoseDetect(float score_thr = 0.25f, float nms_thr = 0.70f, int top_k = 1);
    ~HumanPoseDetect();

    bool ready() const { return m_ready; }
    const char *error() const { return m_error; }
    const std::vector<pose_result_t> &run(const dl::image::img_t &img);
    float last_quality() const { return m_last_quality; }
    int last_valid_count() const { return m_last_valid_count; }
    float last_avg_score() const { return m_last_avg_score; }
    float last_preprocess_ms() const { return m_last_preprocess_ms; }
    float last_forward_ms() const { return m_last_forward_ms; }
    float last_postprocess_ms() const { return m_last_postprocess_ms; }

private:
    dl::Model *m_model = nullptr;
    dl::image::ImagePreprocessor *m_preprocessor = nullptr;
    std::vector<pose_result_t> m_results;
    bool m_ready = false;
    const char *m_error = "Pose model is not ready";
    float m_score_thr;
    float m_nms_thr;
    float m_last_quality = 0.0f;
    float m_last_avg_score = 0.0f;
    float m_last_preprocess_ms = 0.0f;
    float m_last_forward_ms = 0.0f;
    float m_last_postprocess_ms = 0.0f;
    int m_last_valid_count = 0;
    int m_top_k;
    float m_inv_scale_x = 1.0f;
    float m_inv_scale_y = 1.0f;
    int m_border_left = 0;
    int m_border_top = 0;
    int m_crop_left = 0;
    int m_crop_top = 0;
    bool m_profile_pending = true;
};
