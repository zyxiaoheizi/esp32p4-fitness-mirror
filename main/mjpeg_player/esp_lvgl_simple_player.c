/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "esp_log.h"
#include "esp_check.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_cache.h"
#include "esp_private/esp_cache_private.h"
#include "esp_dma_utils.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "driver/jpeg_decode.h"
#include "media_src_storage.h"
#include "bsp/esp-bsp.h"
#include "esp_lvgl_simple_player.h"
#include <inttypes.h>
#include <string.h>

#define CACHE_BUF_ALIGN         (1024)

#define ALIGN_UP(num, align)    (((num) + ((align) - 1)) & ~((align) - 1))
#define ALIGN_DOWN(num, align)    ((num) & ~((align) - 1))

static const char *TAG = "esp_lvgl_player";
static const uint16_t EOI = 0xd9ff; /* End of image */
static TaskHandle_t player_task_handle = NULL;
static volatile bool player_task_running = false;

typedef struct
{
    bool is_init;
    const char              *video_path;
    const char              *bgm_path;
    media_src_t             file;
    uint64_t                filesize;
    jpeg_decoder_handle_t   jpeg;

    uint32_t    screen_width;   /* Width of the video player object */
    uint32_t    screen_height;  /* Height of the video player object */
    uint32_t    video_width;      /* Maximum width of the video */
    uint32_t    video_height;     /* Maximum height of the video  */
    uint32_t    frame_delay_ms;

    player_state_t  state;
    bool            loop;
    bool            hide_controls;
    bool            hide_slider;
    bool            hide_status;
    bool            auto_width;
    bool            auto_height;

    /* Buffers */
    uint8_t     *in_buff;
    uint32_t    in_buff_size;
    uint8_t     *out_buff;
    uint32_t    out_buff_size;
    uint8_t     *canvas_buff;
    uint32_t    canvas_buff_size;
    uint32_t    canvas_width;
    uint32_t    canvas_height;
    SemaphoreHandle_t frame_mutex;
    uint8_t     *cache_buff;
    uint32_t    cache_buff_size;
    bool        cache_buff_in_psram;

    /* LVGL objects */
    lv_obj_t    *main;
    lv_obj_t    *canvas;
    lv_obj_t    *slider;
    lv_obj_t    *btn_play;
    lv_obj_t    *btn_pause;
    lv_obj_t    *btn_stop;
    lv_obj_t    *btn_repeat;
    lv_obj_t    *img_pause;
    lv_obj_t    *img_stop;
    lv_obj_t    *controls;
} player_ctx_t;

static player_ctx_t player_ctx;


static const jpeg_decode_cfg_t jpeg_decode_cfg = {
    .output_format = JPEG_DECODE_OUT_FORMAT_RGB565,
    .rgb_order = JPEG_DEC_RGB_ELEMENT_ORDER_BGR,
};

static uint8_t *find_jpeg_eoi(uint8_t *data, uint32_t len)
{
    if (!data || len < 2) {
        return NULL;
    }
    for (uint32_t i = 0; i + 1 < len; ++i) {
        if (data[i] == 0xff && data[i + 1] == 0xd9) {
            return data + i;
        }
    }
    return NULL;
}



static void play_event_cb(lv_event_t *e)
{
    lv_event_code_t code = lv_event_get_code(e);

    if (code == LV_EVENT_CLICKED) {
        esp_lvgl_simple_player_play();
    }
}

static void stop_event_cb(lv_event_t *e)
{
    lv_event_code_t code = lv_event_get_code(e);

    if (code == LV_EVENT_CLICKED) {
        esp_lvgl_simple_player_stop();

        bsp_display_unlock();
        if (esp_lvgl_simple_player_wait_task_stop(100) != ESP_OK) {
            ESP_LOGE(TAG, "Player task stop timeout");
        }
        bsp_display_lock(100);
    }
}

static void pause_event_cb(lv_event_t *e)
{
    lv_event_code_t code = lv_event_get_code(e);

    if (code == LV_EVENT_CLICKED) {
        if(player_ctx.state == PLAYER_STATE_PAUSED) {
            esp_lvgl_simple_player_play();
        } else if (player_ctx.state == PLAYER_STATE_PLAYING) {
            esp_lvgl_simple_player_pause();
        }
    }
}

static void repeat_event_cb(lv_event_t *e)
{
    lv_event_code_t code = lv_event_get_code(e);
    lv_obj_t *obj = lv_event_get_target(e);

    if (code == LV_EVENT_VALUE_CHANGED) {
        bool loop = lv_obj_get_state(obj) & LV_STATE_CHECKED ? true : false;
        esp_lvgl_simple_player_repeat(loop);
    }
}

static lv_obj_t * create_lvgl_objects(lv_obj_t * screen)
{
    bsp_display_lock(0);

    /* Rows */
    lv_obj_t *cont_col = lv_obj_create(screen);
    lv_obj_remove_style_all(cont_col);
    lv_obj_set_size(cont_col, player_ctx.screen_width, player_ctx.screen_height);
    if (!(player_ctx.hide_controls && player_ctx.hide_slider)) {
        lv_obj_set_flex_flow(cont_col, LV_FLEX_FLOW_COLUMN);
        lv_obj_set_flex_align(cont_col, LV_FLEX_ALIGN_END, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    }
    lv_obj_set_style_pad_all(cont_col, 0, 0);
    lv_obj_set_style_pad_row(cont_col, 0, 0);
    lv_obj_set_style_pad_column(cont_col, 0, 0);
    lv_obj_set_style_bg_color(cont_col, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(cont_col, LV_OPA_COVER, 0);
    lv_obj_set_style_border_width(cont_col, 0, 0);
    lv_obj_set_style_radius(cont_col, 0, 0);
    lv_obj_clear_flag(cont_col, LV_OBJ_FLAG_SCROLLABLE);
    player_ctx.main = cont_col;

    /* Video canvas */
    player_ctx.canvas = lv_canvas_create(cont_col);
    lv_obj_remove_style_all(player_ctx.canvas);
    lv_obj_clear_flag(player_ctx.canvas, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_pos(player_ctx.canvas, 0, 0);
    lv_obj_set_size(player_ctx.canvas, player_ctx.screen_width, player_ctx.screen_height);
    lv_obj_add_event_cb(player_ctx.canvas, pause_event_cb, LV_EVENT_CLICKED, NULL);

    /*Create a slider in the center of the display*/
    lv_obj_t * slider = lv_slider_create(cont_col);
    lv_obj_set_size(slider, player_ctx.screen_width, 5);
    lv_obj_add_state(slider, LV_STATE_DISABLED);
    lv_obj_set_style_opa(slider, LV_OPA_TRANSP, LV_PART_KNOB);
    lv_obj_clear_flag(slider, LV_OBJ_FLAG_CLICKABLE);
    player_ctx.slider = slider;

    /* Buttons */
    lv_obj_t *cont_row = lv_obj_create(cont_col);
    lv_obj_set_size(cont_row, player_ctx.screen_width - 20, 80);
    lv_obj_set_flex_flow(cont_row, LV_FLEX_FLOW_ROW);
    lv_obj_set_style_pad_top(cont_row, 2, 0);
    lv_obj_set_style_pad_bottom(cont_row, 2, 0);
    lv_obj_set_flex_align(cont_row, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER, LV_FLEX_ALIGN_CENTER);
    lv_obj_set_style_bg_color(cont_row, lv_color_black(), 0);
    lv_obj_set_style_border_width(cont_row, 0, 0);
    player_ctx.controls = cont_row;

    /* Play button */
    lv_obj_t * play_btn = lv_btn_create(cont_row);
    lv_obj_t * label = lv_label_create(play_btn);
    lv_label_set_text_static(label, LV_SYMBOL_PLAY);
    lv_obj_add_event_cb(play_btn, play_event_cb, LV_EVENT_CLICKED, NULL);
    player_ctx.btn_play = play_btn;

    /* Pause button */
    lv_obj_t * pause_btn = lv_btn_create(cont_row);
    label = lv_label_create(pause_btn);
    lv_label_set_text_static(label, LV_SYMBOL_PAUSE);
    lv_obj_add_event_cb(pause_btn, pause_event_cb, LV_EVENT_CLICKED, NULL);
    player_ctx.btn_pause = pause_btn;

    /* Stop button */
    lv_obj_t * stop_btn = lv_btn_create(cont_row);
    label = lv_label_create(stop_btn);
    lv_label_set_text_static(label, LV_SYMBOL_STOP);
    lv_obj_add_event_cb(stop_btn, stop_event_cb, LV_EVENT_CLICKED, NULL);
    player_ctx.btn_stop = stop_btn;

    /* Repeat button */
    lv_obj_t * repeat_btn = lv_btn_create(cont_row);
    lv_obj_add_flag(repeat_btn, LV_OBJ_FLAG_CHECKABLE);
    label = lv_label_create(repeat_btn);
    lv_label_set_text_static(label, LV_SYMBOL_REFRESH);
    lv_obj_add_event_cb(repeat_btn, repeat_event_cb, LV_EVENT_VALUE_CHANGED, NULL);
    player_ctx.btn_repeat = repeat_btn;

    /* Pause image */
    lv_obj_t * img_pause = lv_label_create(player_ctx.canvas);
    lv_obj_set_style_text_font(img_pause, &lv_font_montserrat_18, 0);
    lv_obj_set_style_text_color(img_pause, lv_color_white(), 0);
    lv_label_set_text_static(img_pause, LV_SYMBOL_PAUSE);
    lv_obj_center(img_pause);
    lv_obj_add_flag(img_pause, LV_OBJ_FLAG_HIDDEN);
    player_ctx.img_pause = img_pause;

    /* Stop image */
    lv_obj_t * img_stop = lv_label_create(player_ctx.canvas);
    lv_obj_set_style_text_font(img_stop, &lv_font_montserrat_18, 0);
    lv_obj_set_style_text_color(img_stop, lv_color_white(), 0);
    lv_label_set_text_static(img_stop, LV_SYMBOL_STOP);
    lv_obj_center(img_stop);
    lv_obj_add_flag(img_stop, LV_OBJ_FLAG_HIDDEN);
    player_ctx.img_stop = img_stop;

    /* Hide control buttons */
    if (player_ctx.hide_controls) {
        lv_obj_add_flag(cont_row, LV_OBJ_FLAG_HIDDEN);
    }
    /* Hide slider */
    if (player_ctx.hide_slider) {
        lv_obj_add_flag(slider, LV_OBJ_FLAG_HIDDEN);
    }
    /* Hide status icons */
    if (player_ctx.hide_status) {
        lv_obj_add_flag(img_pause, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(img_stop, LV_OBJ_FLAG_HIDDEN);
    }

    bsp_display_unlock();

    return cont_col;
}

static int32_t rescale_video_canvas_locked(void)
{
    if (!player_ctx.canvas || player_ctx.screen_width == 0 || player_ctx.screen_height == 0) {
        return 0;
    }

    lv_image_set_scale(player_ctx.canvas, 256);
    lv_obj_set_size(player_ctx.canvas, player_ctx.screen_width, player_ctx.screen_height);
    if (player_ctx.hide_controls && player_ctx.hide_slider) {
        lv_obj_set_pos(player_ctx.canvas, 0, 0);
    } else {
        lv_obj_center(player_ctx.canvas);
    }
    if (player_ctx.img_pause) {
        lv_obj_center(player_ctx.img_pause);
    }
    if (player_ctx.img_stop) {
        lv_obj_center(player_ctx.img_stop);
    }
    lv_obj_invalidate(player_ctx.canvas);
    return 256;
}

static uint8_t *player_canvas_malloc(uint32_t size, uint32_t *outsize)
{
    const uint32_t aligned = ALIGN_UP(size, 64);
    uint8_t *buf = (uint8_t *)heap_caps_aligned_alloc(64, aligned, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) {
        buf = (uint8_t *)heap_caps_aligned_alloc(64, aligned, MALLOC_CAP_8BIT);
    }
    if (buf && outsize) {
        *outsize = aligned;
    }
    return buf;
}

static bool player_take_frame_mutex(TickType_t ticks)
{
    return !player_ctx.frame_mutex || xSemaphoreTake(player_ctx.frame_mutex, ticks) == pdTRUE;
}

static void player_give_frame_mutex(void)
{
    if (player_ctx.frame_mutex) {
        xSemaphoreGive(player_ctx.frame_mutex);
    }
}

static bool rgb565_near_black(uint16_t c)
{
    const uint16_t r = (c >> 11) & 0x1f;
    const uint16_t g = (c >> 5) & 0x3f;
    const uint16_t b = c & 0x1f;
    return r <= 3 && g <= 6 && b <= 3;
}

static void detect_visible_source_bounds(const uint16_t *src,
                                         uint32_t src_w,
                                         uint32_t src_h,
                                         uint32_t *x,
                                         uint32_t *y,
                                         uint32_t *w,
                                         uint32_t *h)
{
    *x = 0;
    *y = 0;
    *w = src_w;
    *h = src_h;
    if (!src || src_w < 32 || src_h < 32) {
        return;
    }

    const uint32_t row_step = src_w > 480 ? 4 : 2;
    const uint32_t col_step = src_h > 320 ? 4 : 2;
    const uint32_t row_min = src_w / 32 + 2;
    const uint32_t col_min = src_h / 32 + 2;
    uint32_t top = 0;
    uint32_t bottom = src_h - 1;
    uint32_t left = 0;
    uint32_t right = src_w - 1;

    for (uint32_t yy = 0; yy < src_h; ++yy) {
        uint32_t non_black = 0;
        const uint16_t *row = src + yy * src_w;
        for (uint32_t xx = 0; xx < src_w; xx += row_step) {
            if (!rgb565_near_black(row[xx])) {
                non_black++;
            }
        }
        if (non_black >= row_min) {
            top = yy;
            break;
        }
    }

    for (uint32_t yy = src_h; yy > top; --yy) {
        uint32_t non_black = 0;
        const uint16_t *row = src + (yy - 1) * src_w;
        for (uint32_t xx = 0; xx < src_w; xx += row_step) {
            if (!rgb565_near_black(row[xx])) {
                non_black++;
            }
        }
        if (non_black >= row_min) {
            bottom = yy - 1;
            break;
        }
    }

    for (uint32_t xx = 0; xx < src_w; ++xx) {
        uint32_t non_black = 0;
        for (uint32_t yy = top; yy <= bottom; yy += col_step) {
            if (!rgb565_near_black(src[yy * src_w + xx])) {
                non_black++;
            }
        }
        if (non_black >= col_min) {
            left = xx;
            break;
        }
    }

    for (uint32_t xx = src_w; xx > left; --xx) {
        uint32_t non_black = 0;
        for (uint32_t yy = top; yy <= bottom; yy += col_step) {
            if (!rgb565_near_black(src[yy * src_w + (xx - 1)])) {
                non_black++;
            }
        }
        if (non_black >= col_min) {
            right = xx - 1;
            break;
        }
    }

    if (right <= left || bottom <= top) {
        return;
    }

    const uint32_t bw = right - left + 1;
    const uint32_t bh = bottom - top + 1;
    if (bw < src_w / 3 || bh < src_h / 3) {
        return;
    }

    if (left < 4 && top < 4 && src_w - 1 - right < 4 && src_h - 1 - bottom < 4) {
        return;
    }

    *x = left;
    *y = top;
    *w = bw;
    *h = bh;
}

static bool player_reset_canvas_buffer_locked(uint32_t width, uint32_t height)
{
    if (!player_ctx.canvas || width == 0 || height == 0) {
        return false;
    }
    if (player_ctx.canvas_buff && player_ctx.canvas_width == width && player_ctx.canvas_height == height) {
        lv_canvas_set_buffer(player_ctx.canvas, player_ctx.canvas_buff, width, height, LV_COLOR_FORMAT_RGB565);
        return true;
    }

    uint32_t new_size = width * height * 2;
    uint32_t allocated = new_size;
    uint8_t *new_buff = player_canvas_malloc(new_size, &allocated);
    if (!new_buff) {
        ESP_LOGW(TAG, "Resize canvas allocation failed: %" PRIu32 "x%" PRIu32, width, height);
        return false;
    }
    memset(new_buff, 0, new_size);

    uint8_t *old_buff = player_ctx.canvas_buff;
    player_ctx.canvas_buff = new_buff;
    player_ctx.canvas_buff_size = allocated;
    player_ctx.canvas_width = width;
    player_ctx.canvas_height = height;
    lv_canvas_set_buffer(player_ctx.canvas, player_ctx.canvas_buff, width, height, LV_COLOR_FORMAT_RGB565);
    if (old_buff) {
        heap_caps_free(old_buff);
    }
    return true;
}

static void render_video_frame_to_canvas(void)
{
    if (!player_ctx.out_buff || !player_ctx.canvas_buff || player_ctx.video_width == 0 || player_ctx.video_height == 0 ||
        player_ctx.canvas_width == 0 || player_ctx.canvas_height == 0) {
        return;
    }

    if (!player_take_frame_mutex(pdMS_TO_TICKS(30))) {
        return;
    }

    if ((player_ctx.canvas_width != player_ctx.screen_width ||
         player_ctx.canvas_height != player_ctx.screen_height) &&
        player_ctx.screen_width > 0 &&
        player_ctx.screen_height > 0) {
        (void)player_reset_canvas_buffer_locked(player_ctx.screen_width, player_ctx.screen_height);
    }

    const uint32_t src_w = player_ctx.video_width;
    const uint32_t src_h = player_ctx.video_height;
    const uint32_t dst_w = player_ctx.canvas_width;
    const uint32_t dst_h = player_ctx.canvas_height;
    const uint16_t *src = (const uint16_t *)player_ctx.out_buff;
    uint16_t *dst = (uint16_t *)player_ctx.canvas_buff;

    uint32_t crop_x = 0;
    uint32_t crop_y = 0;
    uint32_t crop_w = src_w;
    uint32_t crop_h = src_h;
    detect_visible_source_bounds(src, src_w, src_h, &crop_x, &crop_y, &crop_w, &crop_h);
    const uint32_t base_x = crop_x;
    const uint32_t base_y = crop_y;
    const uint32_t base_w = crop_w;
    const uint32_t base_h = crop_h;

    if ((uint64_t)dst_w * (uint64_t)base_h > (uint64_t)dst_h * (uint64_t)base_w) {
        crop_h = (uint32_t)((uint64_t)base_w * (uint64_t)dst_h / (uint64_t)dst_w);
        crop_h = crop_h == 0 ? 1 : crop_h;
        crop_w = base_w;
        crop_x = base_x;
        crop_y = base_y + (crop_h < base_h ? (base_h - crop_h) / 2 : 0);
    } else {
        crop_w = (uint32_t)((uint64_t)base_h * (uint64_t)dst_w / (uint64_t)dst_h);
        crop_w = crop_w == 0 ? 1 : crop_w;
        crop_h = base_h;
        crop_y = base_y;
        crop_x = base_x + (crop_w < base_w ? (base_w - crop_w) / 2 : 0);
    }

    for (uint32_t y = 0; y < dst_h; ++y) {
        const uint32_t sy = crop_y + (uint32_t)((uint64_t)y * (uint64_t)crop_h / (uint64_t)dst_h);
        const uint16_t *src_row = src + sy * src_w;
        uint16_t *dst_row = dst + y * dst_w;
        for (uint32_t x = 0; x < dst_w; ++x) {
            const uint32_t sx = crop_x + (uint32_t)((uint64_t)x * (uint64_t)crop_w / (uint64_t)dst_w);
            dst_row[x] = src_row[sx];
        }
    }

    player_give_frame_mutex();
}

static esp_err_t get_video_size(uint32_t * width, uint32_t * height)
{
    esp_err_t err;
    jpeg_decode_picture_info_t header;
    assert(width && height);

    int size = media_src_storage_read(&player_ctx.file, player_ctx.cache_buff, player_ctx.cache_buff_size);
    if(size < 0)
        return ESP_ERR_INVALID_SIZE;

    err = jpeg_decoder_get_info(player_ctx.cache_buff, size, &header);

    ESP_LOGI(TAG, "header parsed, width is %" PRId32 ", height is %" PRId32 ", size is %d", header.width, header.height, size);

    *width = header.width;
    *height = header.height;

    return err;
}

static esp_err_t video_decoder_init(void)
{
    jpeg_decode_engine_cfg_t engine_cfg = {
        .intr_priority = 0,
        .timeout_ms = -1,
    };
    return jpeg_new_decoder_engine(&engine_cfg, &player_ctx.jpeg);
}

static void video_decoder_deinit(void)
{
    if (player_ctx.jpeg) {
        jpeg_del_decoder_engine(player_ctx.jpeg);
    }
}

static uint8_t * video_decoder_malloc(uint32_t size, bool inbuff, uint32_t * outsize)
{
    jpeg_decode_memory_alloc_cfg_t tx_mem_cfg = {
        .buffer_direction = JPEG_DEC_ALLOC_INPUT_BUFFER,
    };
    jpeg_decode_memory_alloc_cfg_t rx_mem_cfg = {
        .buffer_direction = JPEG_DEC_ALLOC_OUTPUT_BUFFER,
    };
    return (uint8_t *)jpeg_alloc_decoder_mem(size, (inbuff ? &tx_mem_cfg : &rx_mem_cfg), (size_t*)outsize);
}

static int video_decoder_read_jpeg_image(uint32_t *file_seek_start, uint32_t *file_seek_offset)
{
    int read_size = 0;
    uint32_t jpeg_image_size = 0;
    uint8_t * match = NULL;
    uint8_t *cache_buff = player_ctx.cache_buff;
    uint8_t *cache_buff_offset = NULL;
    uint32_t cache_buff_size = player_ctx.cache_buff_size;
    uint32_t seek_pos_offset = *file_seek_offset;
    uint32_t seek_pos_cur = *file_seek_start;
    uint32_t seek_pos_next = 0;

    // if (seek_pos_cur % CACHE_BUF_ALIGN != 0) {
    //     ESP_LOGE(TAG, "File seek start is not aligned to %d", CACHE_BUF_ALIGN);
    //     return 0;
    // }

    while (match == NULL) {
        int nread = media_src_storage_read(&player_ctx.file, cache_buff, cache_buff_size);
        if (nread < 0) {
            ESP_LOGE(TAG, "Read video file failed");
            return -1;
        }
        read_size = nread - (int)seek_pos_offset;
        if (read_size <= 0) {
            break;
        }

        cache_buff_offset = cache_buff + seek_pos_offset;

        /* Search for EOI. */
        match = find_jpeg_eoi(cache_buff_offset, read_size);
        if(match) {
            read_size = (int)((match + 2) - cache_buff_offset);
        }

        if (jpeg_image_size + (uint32_t)read_size > player_ctx.in_buff_size) {
            ESP_LOGE(TAG,
                     "JPEG frame too large: frame=%" PRIu32 " chunk=%d input=%" PRIu32,
                     jpeg_image_size,
                     read_size,
                     player_ctx.in_buff_size);
            return -1;
        }

        memcpy(player_ctx.in_buff + jpeg_image_size, cache_buff_offset, read_size);
        jpeg_image_size += read_size;

        seek_pos_next = ALIGN_DOWN(seek_pos_cur + seek_pos_offset + read_size, CACHE_BUF_ALIGN);
        seek_pos_cur = seek_pos_cur + seek_pos_offset + read_size;
        seek_pos_offset = seek_pos_cur - seek_pos_next;
        media_src_storage_seek(&player_ctx.file, seek_pos_next);
        seek_pos_cur = seek_pos_next;

    }
    if (jpeg_image_size > player_ctx.in_buff_size) {
        ESP_LOGE(TAG, "JPEG image size is bigger than input buffer size");
        jpeg_image_size = -1;
    }
    *file_seek_start = seek_pos_next;
    *file_seek_offset = seek_pos_offset;

    // if (seek_pos_next % CACHE_BUF_ALIGN != 0) {
    //     ESP_LOGE(TAG, "File seek next is not aligned to %d", CACHE_BUF_ALIGN);
    //     return 0;
    // }

    return jpeg_image_size;
}

static int video_decoder_decode(uint32_t jpeg_image_size)
{
    esp_err_t err;
    uint32_t ret_size = 0;
    uint32_t jpeg_image_size_aligned = ALIGN_UP(jpeg_image_size, 16);

    if (jpeg_image_size_aligned > player_ctx.in_buff_size) {
        ESP_LOGE(TAG, "JPEG image size is bigger than input buffer size");
        return -1;
    }
    if (jpeg_image_size_aligned > jpeg_image_size) {
        memset(player_ctx.in_buff + jpeg_image_size, 0, jpeg_image_size_aligned - jpeg_image_size);
    }

    /* Decode JPEG */
    ret_size = player_ctx.out_buff_size;
    err = jpeg_decoder_process(player_ctx.jpeg, &jpeg_decode_cfg, player_ctx.in_buff, jpeg_image_size_aligned,
                               player_ctx.out_buff, player_ctx.out_buff_size, &ret_size);
    if(err != ESP_OK) {
        ESP_LOGE(TAG, "JPEG decode failed");
        return -1;
    }

    if (ret_size > player_ctx.out_buff_size) {
        ESP_LOGE(TAG, "Output buffer is too small");
        return -1;
    }

    return jpeg_image_size;
}

static void show_video_task(void *arg)
{
    esp_err_t ret = ESP_OK;
    int processed = 0;
    int all_size = 0;
    uint32_t file_seek_pos = 0;
    uint32_t file_seek_offset = 0;
    int jpeg_image_size = 0;

    /* Open video file */
    ESP_LOGI(TAG, "Opening video file %s ...", player_ctx.video_path);
    ESP_GOTO_ON_FALSE(media_src_storage_open(&player_ctx.file) == 0, ESP_ERR_NO_MEM, err, TAG, "Storage open failed");
    ESP_GOTO_ON_FALSE(media_src_storage_connect(&player_ctx.file, player_ctx.video_path) == 0, ESP_ERR_NO_MEM, err, TAG, "Storage connect failed");

    if (player_ctx.bgm_path != NULL) {
        ESP_LOGI(TAG, "Opening bgm file %s ...", player_ctx.bgm_path);
    }

    /* Get file size */
    ESP_GOTO_ON_FALSE(media_src_storage_get_size(&player_ctx.file, &player_ctx.filesize) == 0, ESP_ERR_NO_MEM, err, TAG, "Get file size failed");

    /* Create input buffer */
    player_ctx.in_buff = video_decoder_malloc(player_ctx.in_buff_size, true, &player_ctx.in_buff_size);
    ESP_GOTO_ON_FALSE(player_ctx.in_buff, ESP_ERR_NO_MEM, err, TAG, "Allocation in_buff failed");

    /* Init video decoder */
    ESP_GOTO_ON_ERROR(video_decoder_init(), err, TAG, "Initialize video decoder failed");

    /* Get video output size. JPEG HW decoder aligns both dimensions internally. */
    uint32_t raw_height = 0;
    uint32_t raw_width = 0;
    ESP_GOTO_ON_ERROR(get_video_size(&raw_width, &raw_height), err, TAG, "Get video file size failed");
    uint32_t width = ALIGN_UP(raw_width, 16);
    uint32_t height = ALIGN_UP(raw_height, 16);
    player_ctx.video_width = width;
    player_ctx.video_height = height;

    /* Create output buffer */
    player_ctx.out_buff_size = width * height * 2;
    player_ctx.out_buff = video_decoder_malloc(player_ctx.out_buff_size, false, &player_ctx.out_buff_size);
    ESP_GOTO_ON_FALSE(player_ctx.out_buff, ESP_ERR_NO_MEM, err, TAG, "Allocation out_buff failed");

    player_ctx.canvas_width = player_ctx.screen_width;
    player_ctx.canvas_height = player_ctx.screen_height;
    player_ctx.canvas_buff_size = player_ctx.canvas_width * player_ctx.canvas_height * 2;
    player_ctx.canvas_buff = player_canvas_malloc(player_ctx.canvas_buff_size, &player_ctx.canvas_buff_size);
    ESP_GOTO_ON_FALSE(player_ctx.canvas_buff, ESP_ERR_NO_MEM, err, TAG, "Allocation canvas_buff failed");
    memset(player_ctx.canvas_buff, 0, player_ctx.canvas_width * player_ctx.canvas_height * 2);

    bsp_display_lock(0);
	/* Set buffer to LVGL canvas */
    lv_canvas_set_buffer(player_ctx.canvas,
                         player_ctx.canvas_buff,
                         player_ctx.canvas_width,
                         player_ctx.canvas_height,
                         LV_COLOR_FORMAT_RGB565);
    int32_t scale = rescale_video_canvas_locked();

    if (player_ctx.auto_width || player_ctx.auto_height) {
        uint32_t h = (player_ctx.auto_height ? (height+120) : lv_obj_get_height(player_ctx.main));
        uint32_t w = (player_ctx.auto_width ? width : lv_obj_get_width(player_ctx.main));
        lv_obj_set_size(player_ctx.main, w, h);
    }

    lv_obj_clear_state(player_ctx.slider, LV_STATE_DISABLED);
    /* Enable/disable buttons */
    lv_obj_add_state(player_ctx.btn_play, LV_STATE_DISABLED);
    lv_obj_clear_state(player_ctx.btn_stop, LV_STATE_DISABLED);
    lv_obj_clear_state(player_ctx.btn_pause, LV_STATE_DISABLED);
    lv_obj_clear_state(player_ctx.btn_repeat, LV_STATE_DISABLED);
    /* Hide Stop button */
    lv_obj_add_flag(player_ctx.img_stop, LV_OBJ_FLAG_HIDDEN);
    /* Set slider range */
    lv_slider_set_range(player_ctx.slider, 0, 1000);
    bsp_display_unlock();

    player_ctx.state = PLAYER_STATE_PLAYING;

    ESP_LOGI(TAG,
             "Video player initialized: raw=%" PRIu32 "x%" PRIu32 " aligned=%" PRIu32 "x%" PRIu32 " screen=%" PRIu32 "x%" PRIu32 " scale=%" PRId32,
             raw_width,
             raw_height,
             width,
             height,
             player_ctx.screen_width,
             player_ctx.screen_height,
             scale);

    media_src_storage_seek(&player_ctx.file, 0);
    const TickType_t frame_period = player_ctx.frame_delay_ms > 0 ? pdMS_TO_TICKS(player_ctx.frame_delay_ms) : 0;
    TickType_t frame_wake = xTaskGetTickCount();

    while (player_ctx.state != PLAYER_STATE_STOPPED) {
        if (player_ctx.state == PLAYER_STATE_PAUSED) {
            if (bsp_display_lock(10)) {
                lv_obj_clear_flag(player_ctx.img_pause, LV_OBJ_FLAG_HIDDEN);
                bsp_display_unlock();
            }
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        jpeg_image_size = video_decoder_read_jpeg_image(&file_seek_pos, &file_seek_offset);
        if (jpeg_image_size < 0) {
            ESP_LOGE(TAG, "Read JPEG image failed. Skip frame.");
            break;
        } else if (jpeg_image_size == 0) {
            ESP_LOGI(TAG, "Playing finished.");
            if (player_ctx.loop) {
                ESP_LOGI(TAG, "Playing loop enabled. Play again...");
                media_src_storage_seek(&player_ctx.file, 0);
                file_seek_pos = 0;
                file_seek_offset = 0;
                all_size = 0;
                continue;
            } else {
                esp_lvgl_simple_player_stop();
                continue;
            }
        }

        /* Decode one frame */
        processed = video_decoder_decode(jpeg_image_size);
        if (processed < 0) {
            ESP_LOGE(TAG, "Decode JPEG image failed. Skip frame.");
            break;
        } else {
            all_size += processed;
        }

        render_video_frame_to_canvas();

        if (bsp_display_lock(10)) {
            /* Refresh video canvas object */
            lv_obj_invalidate(player_ctx.canvas);
            /* Set slider */
            lv_slider_set_value(player_ctx.slider, ((float)all_size/(float)player_ctx.filesize)*1000, LV_ANIM_ON);
            bsp_display_unlock();
        }

        if (frame_period > 0) {
            vTaskDelayUntil(&frame_wake, frame_period);
        } else {
            taskYIELD();
        }
    }

err:
    if (bsp_display_lock(10)) {
        /* Show black on screen. Do not block forever here: callers can be inside LVGL events. */
        if (player_ctx.out_buff && player_ctx.out_buff_size > 0) {
            memset(player_ctx.out_buff, 0, player_ctx.out_buff_size);
        }
        if (player_take_frame_mutex(pdMS_TO_TICKS(20))) {
            if (player_ctx.canvas_buff && player_ctx.canvas_width > 0 && player_ctx.canvas_height > 0) {
                memset(player_ctx.canvas_buff, 0, player_ctx.canvas_width * player_ctx.canvas_height * 2);
            }
            player_give_frame_mutex();
        }
        if (player_ctx.auto_height) {
            lv_obj_set_height(player_ctx.main, 320);
        }
        if (player_ctx.canvas) {
            lv_obj_invalidate(player_ctx.canvas);
        }
        if (player_ctx.slider) {
            lv_slider_set_value(player_ctx.slider, 0, LV_ANIM_OFF);
        }
        bsp_display_unlock();
    } else {
        ESP_LOGW(TAG, "Skip final canvas clear because LVGL is busy");
    }

    /* Close storage */
    media_src_storage_disconnect(&player_ctx.file);
    media_src_storage_close(&player_ctx.file);

    /* Deinit video decoder */
    video_decoder_deinit();

    if (player_ctx.in_buff) {
        heap_caps_free(player_ctx.in_buff);
        player_ctx.in_buff = NULL;
    }
    if (player_ctx.out_buff) {
        heap_caps_free(player_ctx.out_buff);
        player_ctx.out_buff = NULL;
        player_ctx.out_buff_size = 0;
    }
    if (player_take_frame_mutex(pdMS_TO_TICKS(100))) {
        if (player_ctx.canvas_buff) {
            heap_caps_free(player_ctx.canvas_buff);
            player_ctx.canvas_buff = NULL;
            player_ctx.canvas_buff_size = 0;
        }
        player_give_frame_mutex();
    }
    player_ctx.canvas_width = 0;
    player_ctx.canvas_height = 0;
    player_ctx.video_width = 0;
    player_ctx.video_height = 0;

    ESP_LOGI(TAG, "Video player task finished.");

    /* Close task */
    player_task_running = false;
    player_task_handle = NULL;
    vTaskDelete(NULL);
}

lv_obj_t * esp_lvgl_simple_player_create(esp_lvgl_simple_player_cfg_t * params)
{
    ESP_RETURN_ON_FALSE(params->video_path, NULL, TAG, "File path must be filled");
    ESP_RETURN_ON_FALSE(params->screen, NULL, TAG, "LVGL screen must be filled");
    ESP_RETURN_ON_FALSE(params->buff_size, NULL, TAG, "Size of the video frame buffer must be filled");
    ESP_RETURN_ON_FALSE(params->screen_width > 0 && params->screen_height > 0, NULL, TAG, "Object size must be filled");

    player_ctx.video_path = params->video_path;
    player_ctx.bgm_path = params->bgm_path;
    player_ctx.in_buff_size = params->buff_size;

    player_ctx.cache_buff_size = ALIGN_UP(params->cache_buff_size, CACHE_BUF_ALIGN);
    player_ctx.cache_buff_in_psram = params->cache_buff_in_psram;
    /* Create split buffer */
    uint32_t flag = player_ctx.cache_buff_in_psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL;
    player_ctx.cache_buff = (uint8_t *)heap_caps_aligned_alloc(128, player_ctx.cache_buff_size, flag);
    if (!player_ctx.cache_buff) {
        ESP_LOGE(TAG, "Malloc cache buffer failed");
        return NULL;
    }

    player_ctx.screen_width = params->screen_width;
    player_ctx.screen_height = params->screen_height;
    player_ctx.video_width = 0;
    player_ctx.video_height = 0;
    player_ctx.canvas_width = 0;
    player_ctx.canvas_height = 0;
    player_ctx.canvas_buff = NULL;
    player_ctx.canvas_buff_size = 0;
    player_ctx.frame_mutex = xSemaphoreCreateMutex();
    if (!player_ctx.frame_mutex) {
        ESP_LOGE(TAG, "Create frame mutex failed");
        if (player_ctx.cache_buff) {
            heap_caps_free(player_ctx.cache_buff);
            player_ctx.cache_buff = NULL;
        }
        return NULL;
    }
    player_ctx.frame_delay_ms = params->frame_delay_ms;
    player_ctx.hide_controls = params->flags.hide_controls;
    player_ctx.hide_slider = params->flags.hide_slider;
    player_ctx.hide_status = params->flags.hide_status;
    player_ctx.auto_width = params->flags.auto_width;
    player_ctx.auto_height = params->flags.auto_height;
    player_ctx.is_init = true;

    /* Create LVGL objects */
    lv_obj_t * player_screen = create_lvgl_objects(params->screen);

    /* Default player state */
    esp_lvgl_simple_player_stop();

    return player_screen;
}

player_state_t esp_lvgl_simple_player_get_state(void)
{
    return player_ctx.state;
}

void esp_lvgl_simple_player_hide_controls(bool hide)
{
    if (!player_ctx.is_init) {
        ESP_LOGW(TAG, "Not init");
        return;
    }

    if (hide) {
        lv_obj_add_flag(player_ctx.controls, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_clear_flag(player_ctx.controls, LV_OBJ_FLAG_HIDDEN);
    }
}

void esp_lvgl_simple_player_change_file(const char *video_file)
{
    if (!player_ctx.is_init) {
        ESP_LOGW(TAG, "Not init");
        return;
    }

    if (player_ctx.state != PLAYER_STATE_STOPPED) {
        ESP_LOGW(TAG, "Playing file can be changed only when video is stopped.");
    }
    player_ctx.video_path = video_file;

    ESP_LOGI(TAG, "Video file changed to %s", video_file);
}

void esp_lvgl_simple_player_play(void)
{
    if (!player_ctx.is_init) {
        ESP_LOGW(TAG, "Not init");
        return;
    }

    if (player_ctx.state == PLAYER_STATE_STOPPED) {
        if (player_task_running) {
            ESP_LOGW(TAG, "Previous player task is still stopping.");
            return;
        }
        ESP_LOGI(TAG, "Player starting playing.");
        /* Create video task */
        player_task_running = true;
        if (xTaskCreate(show_video_task, "video task", 8 * 1024, NULL, 4, &player_task_handle) != pdPASS) {
            player_task_running = false;
            player_task_handle = NULL;
            ESP_LOGE(TAG, "Create video task failed");
        }
    } else if(player_ctx.state == PLAYER_STATE_PAUSED) {
        esp_lvgl_simple_player_resume();
    }
}

void esp_lvgl_simple_player_pause(void)
{
    if (!player_ctx.is_init) {
        ESP_LOGW(TAG, "Not init");
        return;
    }

    if (player_ctx.state == PLAYER_STATE_PLAYING) {
        ESP_LOGI(TAG, "Player paused.");
        player_ctx.state = PLAYER_STATE_PAUSED;
        bsp_display_lock(0);
        lv_obj_clear_state(player_ctx.btn_play, LV_STATE_DISABLED);
        lv_obj_clear_state(player_ctx.btn_stop, LV_STATE_DISABLED);
        lv_obj_clear_state(player_ctx.btn_pause, LV_STATE_DISABLED);
        lv_obj_clear_state(player_ctx.btn_repeat, LV_STATE_DISABLED);
        bsp_display_unlock();
    }
}

void esp_lvgl_simple_player_resume(void)
{
    if (!player_ctx.is_init) {
        ESP_LOGW(TAG, "Not init");
        return;
    }

    if (player_ctx.state == PLAYER_STATE_PAUSED) {
        ESP_LOGI(TAG, "Player resume playing.");
        player_ctx.state = PLAYER_STATE_PLAYING;
        bsp_display_lock(0);
        lv_obj_add_flag(player_ctx.img_pause, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_state(player_ctx.btn_play, LV_STATE_DISABLED);
        bsp_display_unlock();
    }
}

void esp_lvgl_simple_player_stop(void)
{
    if (!player_ctx.is_init) {
        ESP_LOGW(TAG, "Not init");
        return;
    }

    ESP_LOGI(TAG, "Player stopped.");
    player_ctx.state = PLAYER_STATE_STOPPED;

    bsp_display_lock(0);
    lv_obj_clear_state(player_ctx.btn_play, LV_STATE_DISABLED);
    lv_obj_add_state(player_ctx.btn_stop, LV_STATE_DISABLED);
    lv_obj_add_state(player_ctx.btn_pause, LV_STATE_DISABLED);
    lv_obj_add_state(player_ctx.btn_repeat, LV_STATE_DISABLED);
    lv_obj_clear_flag(player_ctx.img_stop, LV_OBJ_FLAG_HIDDEN);
    bsp_display_unlock();
}

void esp_lvgl_simple_player_repeat(bool repeat)
{
    if (!player_ctx.is_init) {
        ESP_LOGW(TAG, "Not init");
        return;
    }

    ESP_LOGI(TAG, "Player repeat %s.", (repeat ? "enabled" : "disabled"));
    player_ctx.loop = repeat;
}

void esp_lvgl_simple_player_resize(uint32_t screen_width, uint32_t screen_height)
{
    if (!player_ctx.is_init || screen_width == 0 || screen_height == 0) {
        return;
    }

    player_ctx.screen_width = screen_width;
    player_ctx.screen_height = screen_height;

    if (bsp_display_lock(50)) {
        if (player_ctx.main) {
            lv_obj_set_size(player_ctx.main, screen_width, screen_height);
        }
        if (player_ctx.slider) {
            lv_obj_set_width(player_ctx.slider, screen_width);
        }
        if (player_ctx.controls) {
            lv_obj_set_width(player_ctx.controls, screen_width > 20 ? screen_width - 20 : screen_width);
        }
        if (player_take_frame_mutex(pdMS_TO_TICKS(80))) {
            (void)player_reset_canvas_buffer_locked(screen_width, screen_height);
            player_give_frame_mutex();
        }
        rescale_video_canvas_locked();
        bsp_display_unlock();
    }
}

esp_err_t esp_lvgl_simple_player_del(void)
{
    if (!player_ctx.is_init) {
        ESP_LOGW(TAG, "Not init");
        return ESP_OK;
    }

    if (player_task_handle != 0) {
        esp_lvgl_simple_player_stop();
        if (esp_lvgl_simple_player_wait_task_stop(-1) != ESP_OK) {
            ESP_LOGE(TAG, "Player task stop timeout");
        }
    }

    if (player_ctx.cache_buff) {
        heap_caps_free(player_ctx.cache_buff);
        player_ctx.cache_buff = NULL;
    }
    if (player_ctx.frame_mutex) {
        vSemaphoreDelete(player_ctx.frame_mutex);
        player_ctx.frame_mutex = NULL;
    }

    player_ctx.is_init = false;

    return ESP_OK;
}

esp_err_t esp_lvgl_simple_player_wait_task_stop(int timeout_ms)
{
    if (!player_ctx.is_init) {
        ESP_LOGW(TAG, "Not init");
        return ESP_OK;
    }

    if (!player_task_running || player_task_handle == 0) {
        player_task_handle = NULL;
        return ESP_OK;
    }

    const int64_t start_us = esp_timer_get_time();
    while (player_task_running) {
        if (timeout_ms >= 0 && esp_timer_get_time() - start_us >= (int64_t)timeout_ms * 1000LL) {
            break;
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    if (player_task_running) {
        ESP_LOGE(TAG, "Player task stop timeout");
        return ESP_ERR_TIMEOUT;
    }

    player_task_handle = NULL;

    return ESP_OK;
}
