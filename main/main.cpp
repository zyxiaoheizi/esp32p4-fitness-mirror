/**
 * ESP32-P4-WIFI6-Touch-LCD-7B camera and face-recognition demo.
 *
 * Notes for this Waveshare board:
 * - The BSP initializes the EK79007 panel and GT911 touch controller.
 * - The BSP LCD and GT911 touch mirror flags are changed together for this
 *   physical 7B mounting. The app itself keeps LVGL rotation at 0 degrees.
 * - Live face recognition runs in a worker task. The camera frame callback only
 *   updates the preview and copies occasional frames, so UI buttons stay usable.
 */

#include <atomic>
#include <algorithm>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <list>
#include <math.h>
#include <new>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#include "sdkconfig.h"

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "esp_cache.h"
#include "esp_private/esp_cache_private.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_event.h"
#include "esp_codec_dev.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "esp_wifi.h"
#include "esp_wifi_remote.h"
#include "lwip/inet.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "lwip/tcp.h"
#include "esp_sntp.h"

#include "bsp/display.h"
#include "bsp/esp-bsp.h"
#include "driver/i2c_master.h"
#include "driver/jpeg_encode.h"
#include "driver/ppa.h"
#include "lvgl.h"

#include "app_video.h"
#include "dl_image_jpeg.hpp"
#include "human_face_detect.hpp"
#include "human_face_recognition.hpp"
#include "human_pose_detect.hpp"
#include "pedestrian_detect.hpp"

#include "mjpeg_player/esp_lvgl_simple_player.h"

static const char *TAG = "face_ui";
extern "C" {
LV_IMAGE_DECLARE(img_competition_logo);
LV_FONT_DECLARE(font_zh_14);
LV_FONT_DECLARE(font_zh_16);
LV_FONT_DECLARE(font_zh_18);
}

enum class UiScreen : int {
    Auth = 0,
    Profile,
    Menu,
    Camera,
    Face,
    Pose,
};

enum class WifiUiState : int {
    Idle = 0,
    Connecting,
    Connected,
    Disconnected,
    Failed,
};

static constexpr int32_t SCREEN_W = BSP_LCD_H_RES;
static constexpr int32_t SCREEN_H = BSP_LCD_V_RES;
static constexpr int32_t PREVIEW_W = (SCREEN_H * 4) / 3;
static constexpr int32_t PREVIEW_H = SCREEN_H;
static constexpr int32_t SIDE_PANEL_X = PREVIEW_W;
static constexpr int32_t SIDE_PANEL_W = SCREEN_W - PREVIEW_W;
static constexpr uint32_t AI_FRAME_W = 640;
static constexpr uint32_t AI_FRAME_H = 480;
static constexpr uint32_t POSE_FRAME_W = 192;
static constexpr uint32_t POSE_FRAME_H = 192;
static constexpr float POSE_OVERLAY_KPT_SCORE_THR = 0.16f;
static constexpr bool YOLO_POSE_USE_FULL_AI_FRAME = true;
static constexpr bool PC_POSE_USE_JPEG = true;
static constexpr bool PC_POSE_USE_RGB332 = false;
static constexpr int PC_POSE_MIN_VISIBLE_KPTS = 4;
static constexpr float PC_POSE_MIN_BOX_RATIO = 0.04f;
static constexpr float PC_POSE_PARSE_SCORE_THR = 0.16f;
static constexpr float PC_POSE_BOX_KPT_SCORE_THR = 0.18f;
static constexpr float PC_POSE_SMOOTH_ALPHA_KPT = 0.88f;
static constexpr float PC_POSE_SMOOTH_ALPHA_BOX = 0.86f;
static constexpr float PC_POSE_LEG_KPT_SCORE_THR = 0.16f;
static constexpr int PC_POSE_MIN_UPPER_KPTS = 2;
static constexpr bool PC_POSE_USE_ADAPTIVE_CROP = true;
static constexpr int64_t PC_POSE_TRACK_CROP_MAX_AGE_US = 520000;
static constexpr float PC_POSE_TRACK_CROP_PAD = 2.45f;
static constexpr float PC_POSE_TRACK_CROP_MIN_RATIO = 0.26f;
static constexpr float PC_POSE_TRACK_KPT_SCORE_THR = 0.14f;
static constexpr int PC_POSE_TRACK_MIN_KPTS = 3;
static constexpr float PC_POSE_ROI_EDGE_MARGIN_RATIO = 0.08f;
static constexpr float PC_POSE_ROI_SEVERE_EDGE_RATIO = 0.03f;
static constexpr float PC_POSE_ROI_DRIFT_RATIO = 0.38f;
static constexpr float PC_POSE_ROI_EDGE_DRIFT_RATIO = 0.18f;
static constexpr int64_t BODY_ROI_DETECT_INTERVAL_US = 260000;
static constexpr float BODY_ROI_DETECT_SCORE_THR = 0.32f;
static constexpr float BODY_ROI_DETECT_NMS_THR = 0.70f;
static constexpr float BODY_ROI_MIN_BOX_RATIO = 0.025f;
static constexpr int LOCAL_POSE_MIN_VISIBLE_KPTS = 6;
static constexpr float LOCAL_POSE_MIN_BOX_RATIO = 0.10f;
static constexpr int POSE_KPT_LEFT_SHOULDER = 11;
static constexpr int POSE_KPT_RIGHT_SHOULDER = 12;
static constexpr int POSE_KPT_LEFT_ELBOW = 13;
static constexpr int POSE_KPT_RIGHT_ELBOW = 14;
static constexpr int POSE_KPT_LEFT_WRIST = 15;
static constexpr int POSE_KPT_RIGHT_WRIST = 16;
static constexpr int POSE_KPT_LEFT_HIP = 23;
static constexpr int POSE_KPT_RIGHT_HIP = 24;
static constexpr int POSE_KPT_LEFT_KNEE = 25;
static constexpr int POSE_KPT_RIGHT_KNEE = 26;
static constexpr int POSE_KPT_LEFT_ANKLE = 27;
static constexpr int POSE_KPT_RIGHT_ANKLE = 28;
static constexpr int POSE_KPT_LEG_START = 23;
static constexpr int64_t PC_POSE_DETAIL_LOG_INTERVAL_US = 1000000;
static constexpr int64_t LIVE_RECOGNITION_INTERVAL_US = 800000;
static constexpr int64_t POSE_REQUEST_INTERVAL_US = 0;
static constexpr int64_t FACE_OVERLAY_TIMEOUT_US = 1500000;
static constexpr int64_t PC_POSE_OVERLAY_HOLD_US = 420000;
static constexpr bool AI_INPUT_ROTATE_180 = true;
static constexpr bool POSE_ENABLE_LOCAL_FALLBACK = false;
static constexpr bool RUN_STATIC_POSE_SELF_TEST = false;
static constexpr bool PC_POSE_USE_UDP = false;
static constexpr int PC_POSE_TCP_TIMEOUT_MS = 320;
static constexpr int PC_POSE_UDP_TIMEOUT_MS = 45;
static constexpr size_t PC_POSE_MAX_UDP_PACKET_BYTES = 1400;
static constexpr int PC_POSE_JPEG_QUALITY = 50;
static constexpr size_t PC_POSE_MAX_JPEG_PAYLOAD_BYTES = 12288;
static constexpr size_t PC_POSE_TCP_SEND_CHUNK_BYTES = 1460;
static constexpr int PC_POSE_TCP_SEND_CHUNK_DELAY_MS = 0;
static constexpr int PC_POSE_SOCKET_BUFFER_BYTES = 8192;
static constexpr int64_t PC_POSE_TCP_FAIL_BACKOFF_US = 100000;
static constexpr int64_t PC_POSE_UDP_FAIL_BACKOFF_US = 3000000;
static constexpr int64_t PC_POSE_UDP_RESET_BACKOFF_US = 100000;
static constexpr int PC_POSE_UDP_MISSES_BEFORE_TCP_FALLBACK = 1;
static constexpr int64_t PC_POSE_FRAME_INTERVAL_US = 10000;
static constexpr int PC_POSE_MISSES_BEFORE_ROI_REACQUIRE = 5;
static constexpr int PC_POSE_MISSES_BEFORE_OVERLAY_CLEAR = 8;
static constexpr int64_t BODY_ROI_SUPPRESS_AFTER_PC_OK_US = 700000;
static constexpr int64_t LOCAL_POSE_INTERVAL_US = 200000;
static constexpr bool LOCAL_POSE_ENABLE = false;
static constexpr bool POSE_INPUT_ROTATE_180 = false;
// PC path: do not rotate the 192x192 JPEG sent to MoveNet; coords map through full-frame letterbox.
static constexpr bool PC_POSE_SEND_ROTATE_180 = false;
static constexpr const char *MUSIC_ROOT = "/sdcard/MUSIC";
static constexpr const char *MUSIC_PLAYLIST_PATH = "/sdcard/MUSIC/playlist.txt";
static constexpr const char *VIDEO_ROOT = "/sdcard/FIT_VIDEO";
static constexpr const char *COURSE_VIDEO_DIR = "/sdcard/FIT_VIDEO/courses";
static constexpr const char *COURSE_VIDEO_MANIFEST = "/sdcard/FIT_VIDEO/courses/manifest.csv";
static constexpr const char *OFFLINE_VIDEO_DIR = "/sdcard/FIT_VIDEO/offline";
static constexpr const char *OFFLINE_VIDEO_MANIFEST = "/sdcard/FIT_VIDEO/offline/manifest.csv";
static constexpr const char *OFFLINE_RESULT_DIR = "/sdcard/FIT_VIDEO/results";
static constexpr const char *RECORD_VIDEO_DIR = "/sdcard/FIT_VIDEO/recordings";
static constexpr uint32_t RECORD_VIDEO_W = 384;
static constexpr uint32_t RECORD_VIDEO_H = 216;
static constexpr int RECORD_VIDEO_FPS = 8;
static constexpr int RECORD_VIDEO_SECONDS = 15;
static constexpr int OFFLINE_ANALYSIS_SAMPLE_FPS = 6;
static constexpr int OFFLINE_ANALYSIS_MIN_SAMPLES = 54;
static constexpr int OFFLINE_ANALYSIS_MAX_SAMPLES = 90;
static constexpr int TRAINING_HISTORY_CAPACITY = 24;
static constexpr int MUSIC_DEFAULT_VOLUME = 58;
static constexpr size_t MUSIC_IO_BUFFER_BYTES = 8192;

extern const uint8_t test_person_jpg_start[] asm("_binary_test_person_jpg_start");
extern const uint8_t test_person_jpg_end[] asm("_binary_test_person_jpg_end");

static std::atomic<int> g_ui_screen{static_cast<int>(UiScreen::Auth)};
static lv_display_t *s_disp = nullptr;
static bool s_net_stack_ready = false;

static HumanFaceDetect *s_det = nullptr;
static HumanFaceRecognizer *s_rec = nullptr;
static HumanPoseDetect *s_pose_det = nullptr;
static PedestrianDetect *s_body_det = nullptr;

static lv_obj_t *scr_menu = nullptr;
static lv_obj_t *scr_boot = nullptr;
static lv_obj_t *scr_auth = nullptr;
static lv_obj_t *scr_profile = nullptr;
static lv_obj_t *scr_cam = nullptr;
static lv_obj_t *scr_face = nullptr;
static lv_obj_t *scr_pose = nullptr;
static lv_obj_t *scr_start = nullptr;
static lv_obj_t *scr_correction = nullptr;
static lv_obj_t *scr_courses = nullptr;
static lv_obj_t *scr_plan = nullptr;
static lv_obj_t *scr_report = nullptr;
static lv_obj_t *scr_settings = nullptr;
static lv_obj_t *scr_music = nullptr;
static lv_obj_t *cv_auth = nullptr;
static lv_obj_t *cv_cam = nullptr;
static lv_obj_t *cv_face = nullptr;
static lv_obj_t *cv_pose = nullptr;
static lv_obj_t *view_cam = nullptr;
static lv_obj_t *view_auth = nullptr;
static lv_obj_t *view_face = nullptr;
static lv_obj_t *view_pose = nullptr;
static lv_obj_t *lbl_status = nullptr;
static lv_obj_t *lbl_pose_status = nullptr;
static lv_obj_t *box_face = nullptr;
static lv_obj_t *lbl_face_name = nullptr;
static lv_obj_t *lbl_db_count = nullptr;
static lv_obj_t *ta_name = nullptr;
static lv_obj_t *ta_del_id = nullptr;
static lv_obj_t *kb_name = nullptr;
static lv_obj_t *sw_live = nullptr;
static lv_obj_t *lbl_auth_status = nullptr;
static lv_obj_t *s_auth_brand = nullptr;
static lv_obj_t *ta_auth_name = nullptr;
static lv_obj_t *ta_auth_pass = nullptr;
static lv_obj_t *kb_auth = nullptr;
static lv_obj_t *ta_profile_name = nullptr;
static lv_obj_t *ta_profile_height = nullptr;
static lv_obj_t *ta_profile_weight = nullptr;
static lv_obj_t *ta_profile_minutes = nullptr;
static lv_obj_t *lbl_profile_choice = nullptr;
static lv_obj_t *lbl_profile_status = nullptr;
static lv_obj_t *kb_profile = nullptr;
static lv_obj_t *lbl_wifi_state = nullptr;
static lv_obj_t *lbl_wifi_ip = nullptr;
static lv_obj_t *lbl_wifi_saved_ssid = nullptr;
static lv_obj_t *lbl_settings_account = nullptr;
static lv_obj_t *lbl_settings_face = nullptr;
static lv_obj_t *lbl_settings_profile = nullptr;
static lv_obj_t *lbl_settings_heap = nullptr;
static lv_obj_t *lbl_settings_uptime = nullptr;
static lv_obj_t *lbl_settings_pc = nullptr;
static lv_obj_t *lbl_menu_wifi = nullptr;
static lv_obj_t *lbl_menu_pc = nullptr;
static lv_obj_t *lbl_menu_ip = nullptr;
static lv_obj_t *lbl_menu_music = nullptr;
static lv_obj_t *lbl_menu_datetime = nullptr;
static lv_obj_t *lbl_menu_weather = nullptr;
static lv_obj_t *ta_wifi_ssid = nullptr;
static lv_obj_t *ta_wifi_pass = nullptr;
static lv_obj_t *kb_settings = nullptr;
static lv_obj_t *lbl_music_title = nullptr;
static lv_obj_t *lbl_music_artist = nullptr;
static lv_obj_t *lbl_music_status = nullptr;
static lv_obj_t *lbl_music_time = nullptr;
static lv_obj_t *lbl_music_cover_title = nullptr;
static lv_obj_t *lbl_music_cover_path = nullptr;
static lv_obj_t *img_music_cover = nullptr;
static lv_obj_t *bar_music_progress = nullptr;
static lv_obj_t *slider_music_volume = nullptr;
static lv_obj_t *slider_menu_volume = nullptr;
static lv_obj_t *btn_music_play_label = nullptr;
static lv_obj_t *lbl_pose_music = nullptr;
static lv_obj_t *lbl_train_subtitle = nullptr;
static lv_obj_t *lbl_train_count = nullptr;
static lv_obj_t *lbl_train_timer = nullptr;
static lv_obj_t *lbl_train_score = nullptr;
static lv_obj_t *lbl_train_state = nullptr;
static lv_obj_t *lbl_train_cue = nullptr;
static lv_obj_t *lbl_corr_summary = nullptr;
static lv_obj_t *lbl_corr_depth = nullptr;
static lv_obj_t *lbl_corr_knee = nullptr;
static lv_obj_t *lbl_corr_hip = nullptr;
static lv_obj_t *lbl_corr_torso = nullptr;
static lv_obj_t *lbl_corr_balance = nullptr;
static lv_obj_t *lbl_corr_track = nullptr;
static lv_obj_t *lbl_corr_tip = nullptr;
static lv_obj_t *btn_train_primary_label = nullptr;
static lv_obj_t *lbl_start_current = nullptr;
static lv_obj_t *lbl_start_target = nullptr;
static lv_obj_t *lbl_start_focus = nullptr;
static lv_obj_t *lbl_start_sets_value = nullptr;
static lv_obj_t *lbl_start_reps_title = nullptr;
static lv_obj_t *lbl_start_reps_value = nullptr;
static lv_obj_t *lbl_start_guide = nullptr;
static lv_obj_t *panel_plan_today = nullptr;
static lv_obj_t *panel_plan_week = nullptr;
static lv_obj_t *btn_plan_week_label = nullptr;
static lv_obj_t *s_course_list = nullptr;
static lv_obj_t *s_course_player_card = nullptr;
static lv_obj_t *s_course_player_host = nullptr;
static lv_obj_t *s_course_title_label = nullptr;
static lv_obj_t *s_course_status_label = nullptr;
static lv_obj_t *s_course_zoom_button = nullptr;
static lv_obj_t *s_offline_list = nullptr;
static lv_obj_t *s_offline_player_card = nullptr;
static lv_obj_t *s_offline_player_host = nullptr;
static lv_obj_t *s_offline_title_label = nullptr;
static lv_obj_t *s_offline_status_label = nullptr;
static lv_obj_t *s_offline_report_label = nullptr;
static lv_obj_t *s_offline_progress = nullptr;
static lv_obj_t *s_offline_zoom_button = nullptr;

static int s_vfd = -1;
static bool s_streaming = false;
static uint8_t *s_cam_buf[EXAMPLE_CAM_BUF_NUM] = {};
static size_t s_cam_buf_bytes = 0;
static uint32_t s_cam_w = 0;
static uint32_t s_cam_h = 0;
static size_t s_cache_line = 0;

static uint8_t *s_snap_buf = nullptr;
static uint8_t *s_live_buf = nullptr;
static uint8_t *s_ai_buf = nullptr;
static uint8_t *s_pose_buf = nullptr;
static uint8_t *s_body_det_buf = nullptr;
static uint8_t *s_preview_buf = nullptr;
static uint32_t s_ai_w = 0;
static uint32_t s_ai_h = 0;
static uint32_t s_pose_w = 0;
static uint32_t s_pose_h = 0;
static std::atomic<bool> s_snap_request{false};
static std::atomic<bool> s_live_busy{false};
static std::atomic<bool> s_enroll_busy{false};
static std::atomic<bool> s_enroll_cancel{false};
static std::atomic<bool> s_pose_busy{false};
static std::atomic<bool> s_body_roi_busy{false};
static std::atomic<int> s_db_count{0};
static std::atomic<int64_t> s_last_recognition_us{0};
static std::atomic<int64_t> s_last_pose_request_us{0};
static std::atomic<int64_t> s_last_body_roi_request_us{0};
static std::atomic<bool> s_body_roi_reacquire_requested{false};
static std::atomic<int64_t> s_last_pc_pose_success_us{0};
static std::atomic<int64_t> s_last_preview_update_us{0};
static std::atomic<bool> s_preview_dirty{false};
static SemaphoreHandle_t s_snap_sem = nullptr;
static SemaphoreHandle_t s_live_sem = nullptr;
static SemaphoreHandle_t s_enroll_sem = nullptr;
static SemaphoreHandle_t s_pose_sem = nullptr;
static SemaphoreHandle_t s_body_roi_sem = nullptr;
static SemaphoreHandle_t s_model_mutex = nullptr;
static SemaphoreHandle_t s_overlay_mutex = nullptr;
static SemaphoreHandle_t s_pose_overlay_mutex = nullptr;
static TaskHandle_t s_worker_task = nullptr;
static EventGroupHandle_t s_pc_pose_wifi_events = nullptr;
static esp_netif_t *s_pc_pose_netif = nullptr;
static int s_frame_ctr = 0;
static bool s_live_rec = false;
static bool s_pose_enabled = false;
static char s_enroll_name[32] = {};
static char s_wifi_ssid[33] = {};
static char s_wifi_password[65] = {};
static char s_wifi_ip[16] = {};
static char s_weather_text[48] = "天气: 待接入";
static char s_auth_saved_user[32] = {};
static char s_auth_saved_pass[64] = {};
static char s_auth_user[32] = {};
static int s_face_input_variant = -1;
static int s_pose_input_variant = -1;
static std::atomic<int> s_wifi_ui_state{0};
static std::atomic<bool> s_auth_has_password{false};
static std::atomic<bool> s_auth_face_login_active{false};
static std::atomic<bool> s_auth_unlock_pending{false};
static std::atomic<bool> s_pc_pose_wifi_ready{false};
static std::atomic<bool> s_pc_pose_wifi_started{false};
static std::atomic<bool> s_time_sync_started{false};
static TaskHandle_t s_wifi_reconnect_task = nullptr;
static TaskHandle_t s_wifi_watchdog_task = nullptr;
static std::atomic<int64_t> s_wifi_connect_started_us{0};
static std::atomic<int64_t> s_wifi_last_recover_us{0};
static int s_pc_pose_retry = 0;
static uint8_t *s_pc_pose_resp_buf = nullptr;
static size_t s_pc_pose_resp_cap = 0;
static uint8_t *s_pc_pose_tx_buf = nullptr;
static size_t s_pc_pose_tx_cap = 0;
static int s_pc_pose_sock = -1;
static int s_pc_pose_udp_sock = -1;
static uint32_t s_pc_pose_frame_seq = 0;
static uint32_t s_pc_voice_seq = 0;
static uint32_t s_weather_seq = 0;
static std::atomic<int64_t> s_pc_pose_next_connect_us{0};
static std::atomic<int64_t> s_pc_pose_udp_retry_after_us{0};
static uint8_t *s_pc_pose_buf = nullptr;
static uint8_t *s_pc_pose_packet_buf = nullptr;
static size_t s_pc_pose_packet_cap = 0;
static jpeg_encoder_handle_t s_pc_pose_jpeg_handle = nullptr;
static uint8_t *s_pc_pose_jpeg_out = nullptr;
static size_t s_pc_pose_jpeg_out_cap = 0;
static bool s_pc_pose_jpeg_hw_disabled = false;
static ppa_client_handle_t s_pc_pose_ppa_srm = nullptr;
static bool s_pc_pose_ppa_disabled = false;
static SemaphoreHandle_t s_pc_pose_sem = nullptr;
static TaskHandle_t s_pc_pose_task = nullptr;
static std::atomic<bool> s_pc_pose_busy{false};
static std::atomic<int64_t> s_last_pc_pose_request_us{0};
static uint32_t s_pc_pose_crop_x = 0;
static uint32_t s_pc_pose_crop_y = 0;
static uint32_t s_pc_pose_crop_w = 0;
static uint32_t s_pc_pose_crop_h = 0;
static bool s_pc_pose_use_crop = false;
static uint32_t s_pc_pose_letterbox_x = 0;
static uint32_t s_pc_pose_letterbox_y = 0;
static uint32_t s_pc_pose_letterbox_w = 0;
static uint32_t s_pc_pose_letterbox_h = 0;
static bool s_pc_pose_use_letterbox = false;
static uint32_t s_local_pose_letterbox_x = 0;
static uint32_t s_local_pose_letterbox_y = 0;
static uint32_t s_local_pose_letterbox_w = 0;
static uint32_t s_local_pose_letterbox_h = 0;
static bool s_local_pose_use_letterbox = false;
static uint32_t s_local_pose_crop_x = 0;
static uint32_t s_local_pose_crop_y = 0;
static uint32_t s_local_pose_crop_w = 0;
static uint32_t s_local_pose_crop_h = 0;
static bool s_local_pose_use_crop = false;

typedef struct {
    bool valid;
    uint32_t x;
    uint32_t y;
    uint32_t w;
    uint32_t h;
    int64_t ts_us;
} pose_input_roi_t;

static pose_input_roi_t s_pose_track_roi = {};
static portMUX_TYPE s_pose_roi_lock = portMUX_INITIALIZER_UNLOCKED;

typedef struct {
    const char *name;
    dl::image::pix_type_t pix_type;
    bool rotate_180;
} face_input_variant_t;

typedef struct {
    const char *name;
    dl::image::pix_type_t pix_type;
} pose_input_variant_t;

static constexpr face_input_variant_t FACE_INPUT_VARIANTS[] = {
    {"RGB565LE", dl::image::DL_IMAGE_PIX_TYPE_RGB565LE, false},
    {"RGB565BE", dl::image::DL_IMAGE_PIX_TYPE_RGB565BE, false},
    {"BGR565LE", dl::image::DL_IMAGE_PIX_TYPE_BGR565LE, false},
    {"BGR565BE", dl::image::DL_IMAGE_PIX_TYPE_BGR565BE, false},
    {"RGB565LE_R180", dl::image::DL_IMAGE_PIX_TYPE_RGB565LE, true},
    {"RGB565BE_R180", dl::image::DL_IMAGE_PIX_TYPE_RGB565BE, true},
    {"BGR565LE_R180", dl::image::DL_IMAGE_PIX_TYPE_BGR565LE, true},
    {"BGR565BE_R180", dl::image::DL_IMAGE_PIX_TYPE_BGR565BE, true},
};

static constexpr pose_input_variant_t POSE_INPUT_VARIANTS[] = {
    {"RGB565LE", dl::image::DL_IMAGE_PIX_TYPE_RGB565LE},
    {"RGB565BE", dl::image::DL_IMAGE_PIX_TYPE_RGB565BE},
    {"BGR565LE", dl::image::DL_IMAGE_PIX_TYPE_BGR565LE},
    {"BGR565BE", dl::image::DL_IMAGE_PIX_TYPE_BGR565BE},
};

typedef struct {
    bool visible;
    int x1;
    int y1;
    int x2;
    int y2;
    int64_t ts_us;
    char text[80];
} face_overlay_t;

static face_overlay_t s_face_overlay = {};

static constexpr int POSE_MAX_PEOPLE = 4;
static constexpr int POSE_KPTS = 33;
static constexpr int LOCAL_POSE_FRAME_W = 192;
static constexpr int LOCAL_POSE_FRAME_H = 192;

typedef struct {
    bool valid;
    int x;
    int y;
} pose_overlay_point_t;

typedef struct {
    bool visible;
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    pose_overlay_point_t kpts[POSE_KPTS];
} pose_overlay_person_t;

typedef struct {
    bool visible;
    int count;
    int64_t ts_us;
    pose_overlay_person_t people[POSE_MAX_PEOPLE];
} pose_overlay_t;

static pose_overlay_t s_pose_overlay_local = {};
static pose_overlay_t s_pose_overlay_pc = {};
static pose_overlay_t s_pc_pose_overlay_hold = {};
static pose_result_t s_pc_pose_smooth_prev = {};
static bool s_pc_pose_smooth_ready = false;
static pose_result_t s_offline_analysis_smooth_prev = {};
static bool s_offline_analysis_smooth_ready = false;
static uint8_t s_offline_analysis_kpt_miss[POSE_KPTS] = {};
static pose_result_t s_offline_draw_smooth_prev = {};
static bool s_offline_draw_smooth_ready = false;

static const char *POSE_KEYPOINT_NAMES[POSE_KPTS] = {
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
};

typedef struct {
    char line[160];
} ui_msg_t;

typedef struct {
    std::string id;
    std::string title;
    std::string audio_path;
    std::string cover_path;
    uint32_t duration_ms;
} music_track_t;

typedef struct {
    std::string id;
    std::string title;
    std::string action;
    std::string path;
} video_item_t;

enum class MusicCmdType : uint8_t {
    PlayIndex,
    Toggle,
    Stop,
    Next,
    Prev,
    SetVolume,
    Rescan,
    VoicePrompt,
};

enum class VoicePromptId : int {
    BadDepth = 0,
    BadTorso,
    BadKnee,
    BadBalance,
    GoodStreak,
    TrainingComplete,
    PushupDepth,
    PushupBodyLine,
    BenchRange,
    BenchShoulder,
    PullupHeight,
    PullupSwing,
    DeadliftBack,
    DeadliftHip,
    PlankHip,
    PlankHold,
    PushupHip,
    PushupElbow,
    BenchWrist,
    PullupGrip,
    PullupControl,
    DeadliftKnee,
    PlankRestart,
    PlankSag,
    CurlRange,
    CurlSwing,
    SitupRange,
    SitupNeck,
};

typedef struct {
    MusicCmdType type;
    int value;
} music_cmd_t;

typedef struct {
    uint8_t *data;
    size_t len;
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t bits_per_sample;
} voice_pcm_t;

typedef struct {
    bool sd_ready;
    bool library_ready;
    bool playing;
    bool paused;
    bool error;
    int index;
    int volume;
    uint32_t position_ms;
    uint32_t duration_ms;
    char title[64];
    char artist[64];
    char audio_path[160];
    char cover_path[160];
    char status[96];
} music_state_t;

enum class TrainingRunState : int {
    Ready = 0,
    Running,
    Paused,
    Finished,
};

typedef struct {
    const char *name;
    const char *meta;
    const char *target;
    const char *focus;
    uint8_t sets;
    uint8_t reps_per_set;
    uint16_t duration_sec;
    bool counted;
    uint32_t color;
} training_profile_t;

static constexpr training_profile_t TRAINING_PROFILES[] = {
    {"深蹲", "4组 x 12次 - 膝盖轨迹", "4组 x 12次", "膝盖轨迹、髋部深度、躯干稳定", 4, 12, 0, true, 0x145883},
    {"俯卧撑", "3组 x 10次 - 身体线条", "3组 x 10次", "肩髋保持一条线，下降速度稳定", 3, 10, 0, true, 0x2f7dd1},
    {"卧推", "4组 x 8次 - 推起控制", "4组 x 8次", "手腕肘肩同步，动作幅度完整", 4, 8, 0, true, 0x22a06b},
    {"引体向上", "4组 x 6次 - 背部发力", "4组 x 6次", "身体稳定，胸口主动靠近横杠", 4, 6, 0, true, 0xd9822b},
    {"硬拉", "4组 x 8次 - 髋部发力", "4组 x 8次", "背部稳定，髋膝协调伸展", 4, 8, 0, true, 0x6f65c8},
    {"平板支撑", "3组 x 45秒 - 核心", "45秒保持", "肩髋踝一条线，避免塌腰和抬臀", 3, 0, 45, false, 0xd64545},
    {"哑铃弯举", "3组 x 12次 - 上臂稳定", "3组 x 12次", "上臂贴近身体，完整屈肘并控制下放", 3, 12, 0, true, 0x0f8a8a},
    {"仰卧起坐", "3组 x 15次 - 腹部发力", "3组 x 15次", "腹部卷起，控制下放，不要猛拉颈部", 3, 15, 0, true, 0x8a5cf6},
};

static constexpr int TRAINING_PROFILE_COUNT = (int)(sizeof(TRAINING_PROFILES) / sizeof(TRAINING_PROFILES[0]));
static constexpr uint8_t TRAINING_GOAL_VERSION = 3;

typedef struct {
    const training_profile_t *profile;
    TrainingRunState state;
    int count;
    int target_count;
    int current_set;
    uint32_t elapsed_ms;
    int score;
    int score_depth;
    int score_knee;
    int score_hip;
    int score_torso;
    int score_balance;
    int score_track;
    float knee_angle;
    float hip_angle;
    float torso_lean;
    float depth_ratio;
    float symmetry_delta;
    int valid_kpts;
    float pc_fps;
    char cue[120];
    char detail[160];
} training_snapshot_t;

typedef struct {
    bool complete;
    char name[32];
    char gender[16];
    char goal[24];
    char focus[24];
    uint16_t height_cm;
    uint16_t weight_kg;
    uint16_t minutes_per_day;
} user_profile_t;

typedef struct {
    uint32_t sessions;
    int last_score;
    int last_score_depth;
    int last_score_knee;
    int last_score_hip;
    int last_score_torso;
    int last_score_balance;
    int last_score_track;
    int last_count;
    int last_target;
    uint32_t last_elapsed_ms;
    uint32_t total_count;
    uint32_t total_elapsed_ms;
    char last_training[32];
    char last_time[32];
    char last_tip[96];
} training_record_t;

typedef struct {
    bool valid;
    char source[12];
    char action[32];
    char time[32];
    char tip[120];
    char reps[180];
    int count;
    int target;
    int score;
    int rep_score;
    int score_depth;
    int score_knee;
    int score_hip;
    int score_torso;
    int score_balance;
    int score_track;
    uint32_t elapsed_ms;
} training_history_item_t;

typedef struct {
    int samples;
    float confidence_sum;
    int valid_kpts_sum;
    float max_depth;
    float min_knee_angle;
    bool has_knee_angle;
    float max_knee_track_delta;
    bool has_knee_track;
    float min_hip_angle;
    float max_torso_lean;
    float max_symmetry_delta;
    int score;
    int score_depth;
    int score_knee;
    int score_hip;
    int score_torso;
    int score_balance;
    int score_track;
    VoicePromptId voice;
    bool needs_voice;
    bool good;
    char detail[160];
} squat_rep_eval_t;

static QueueHandle_t s_uiq = nullptr;
static QueueHandle_t s_music_cmd_q = nullptr;
static SemaphoreHandle_t s_music_mutex = nullptr;
static SemaphoreHandle_t s_weather_mutex = nullptr;
static SemaphoreHandle_t s_training_mutex = nullptr;
static TaskHandle_t s_music_task = nullptr;
static TaskHandle_t s_weather_task = nullptr;
static TaskHandle_t s_sdcard_retry_task = nullptr;
static bool s_lvgl_build_failed = false;
static bool s_sdcard_mounted = false;
static bool s_music_audio_ready = false;
static esp_codec_dev_handle_t s_music_codec = nullptr;
static std::vector<music_track_t> s_music_tracks;
static std::vector<video_item_t> s_course_videos;
static std::vector<video_item_t> s_offline_videos;
static int s_course_selected_index = -1;
static int s_offline_selected_index = -1;
static bool s_course_video_zoomed = false;
static bool s_offline_video_zoomed = false;
static TaskHandle_t s_offline_process_task = nullptr;
static lv_obj_t *s_video_player_obj = nullptr;
static char s_video_player_path[192] = {};
static music_state_t s_music_state = {
    .sd_ready = false,
    .library_ready = false,
    .playing = false,
    .paused = false,
    .error = false,
    .index = -1,
    .volume = MUSIC_DEFAULT_VOLUME,
    .position_ms = 0,
    .duration_ms = 0,
    .title = "--",
    .artist = "本地音乐",
    .audio_path = "",
    .cover_path = "",
    .status = "等待SD卡",
};

static int s_training_index = 0;
static uint8_t s_training_goal_sets[TRAINING_PROFILE_COUNT] = {};
static uint8_t s_training_goal_reps[TRAINING_PROFILE_COUNT] = {};
static uint16_t s_training_goal_duration_sec[TRAINING_PROFILE_COUNT] = {};
static TrainingRunState s_training_state = TrainingRunState::Ready;
static int s_training_count = 0;
static int s_training_current_set = 1;
static int64_t s_training_start_us = 0;
static int64_t s_training_elapsed_us = 0;
static bool s_training_phase_active = false;
static int s_training_stand_frames = 0;
static int s_training_down_frames = 0;
static int s_training_up_frames = 0;
static int s_training_rep_cooldown_frames = 0;
static int64_t s_training_last_rep_us = 0;
static int64_t s_training_phase_start_us = 0;
static bool s_training_needs_rearm = false;
static float s_training_motion_ref = 0.0f;
static float s_training_phase_peak_signal = 0.0f;
static int s_training_hold_good_frames = 0;
static int s_training_hold_bad_frames = 0;
static int s_training_score = 0;
static int s_training_score_depth = 0;
static int s_training_score_knee = 0;
static int s_training_score_hip = 0;
static int s_training_score_torso = 0;
static int s_training_score_balance = 0;
static int s_training_score_track = 0;
static int s_training_session_score_sum = 0;
static int s_training_session_depth_sum = 0;
static int s_training_session_knee_sum = 0;
static int s_training_session_hip_sum = 0;
static int s_training_session_torso_sum = 0;
static int s_training_session_balance_sum = 0;
static int s_training_session_track_sum = 0;
static int s_training_scored_reps = 0;
static int s_training_good_rep_streak = 0;
static squat_rep_eval_t s_training_current_rep = {};
static squat_rep_eval_t s_training_last_rep = {};
static bool s_training_simple_rep_valid = false;
static int s_training_simple_rep_depth = 0;
static int s_training_simple_rep_knee = 0;
static int s_training_simple_rep_hip = 0;
static int s_training_simple_rep_torso = 0;
static int s_training_simple_rep_balance = 0;
static int s_training_simple_rep_track = 0;
static float s_training_knee_angle = 0.0f;
static float s_training_hip_angle = 0.0f;
static float s_training_torso_lean = 0.0f;
static float s_training_depth_ratio = 0.0f;
static float s_training_symmetry_delta = 0.0f;
static int s_training_valid_kpts = 0;
static float s_training_pc_fps = 0.0f;
static int64_t s_training_last_pose_us = 0;
static int64_t s_training_last_debug_us = 0;
static VoicePromptId s_training_last_voice_prompt = VoicePromptId::TrainingComplete;
static int64_t s_training_last_voice_us = 0;
static char s_training_cue[120] = "选择训练项目后开始";
static char s_training_detail[160] = "开始训练后显示分项评分。";
static lv_obj_t *s_record_button = nullptr;
static lv_obj_t *s_record_status_label = nullptr;
static SemaphoreHandle_t s_record_mutex = nullptr;
static FILE *s_record_fp = nullptr;
static uint8_t *s_record_frame_buf = nullptr;
static bool s_record_armed = false;
static bool s_recording = false;
static int64_t s_record_start_us = 0;
static int64_t s_record_last_frame_us = 0;
static int s_record_frames = 0;
static char s_record_action[24] = "squat";
static char s_record_path[192] = {};
static user_profile_t s_user_profile = {};
static training_record_t s_training_record = {};
static training_history_item_t s_training_history[TRAINING_HISTORY_CAPACITY] = {};
static int s_training_history_count = 0;
static bool s_training_record_saved = false;
static char s_profile_gender_sel[16] = "男";
static char s_profile_goal_sel[24] = "塑形";
static char s_profile_focus_sel[24] = "全身";

static void push_ui_msg(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static const lv_font_t *font_14(void);
static const lv_font_t *font_16(void);
static const lv_font_t *font_18(void);
static lv_obj_t *make_text(lv_obj_t *parent,
                           const char *text,
                           int32_t x,
                           int32_t y,
                           int32_t w,
                           const lv_font_t *font,
                           uint32_t color,
                           lv_label_long_mode_t mode);

static void mark_lvgl_build_failed(const char *what)
{
    s_lvgl_build_failed = true;
    ESP_LOGW(TAG, "LVGL object create failed: %s", what ? what : "unknown");
}

static const char *reset_reason_name(esp_reset_reason_t reason)
{
    switch (reason) {
    case ESP_RST_POWERON:
        return "POWERON";
    case ESP_RST_EXT:
        return "EXT";
    case ESP_RST_SW:
        return "SW";
    case ESP_RST_PANIC:
        return "PANIC";
    case ESP_RST_INT_WDT:
        return "INT_WDT";
    case ESP_RST_TASK_WDT:
        return "TASK_WDT";
    case ESP_RST_WDT:
        return "WDT";
    case ESP_RST_DEEPSLEEP:
        return "DEEPSLEEP";
    case ESP_RST_BROWNOUT:
        return "BROWNOUT";
    case ESP_RST_SDIO:
        return "SDIO";
    default:
        return "UNKNOWN";
    }
}

static void notify_worker(void)
{
    if (s_worker_task) {
        xTaskNotifyGive(s_worker_task);
    }
}

static void notify_pc_pose_worker(void)
{
    if (s_pc_pose_task) {
        xTaskNotifyGive(s_pc_pose_task);
    }
}

static void push_ui_msg(const char *fmt, ...)
{
    if (!s_uiq) {
        return;
    }

    ui_msg_t msg = {};
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg.line, sizeof(msg.line), fmt, ap);
    va_end(ap);
    xQueueSend(s_uiq, &msg, 0);
}

static std::string trim_copy(const std::string &in)
{
    size_t first = 0;
    while (first < in.size() && isspace((unsigned char)in[first])) {
        first++;
    }
    size_t last = in.size();
    while (last > first && isspace((unsigned char)in[last - 1])) {
        last--;
    }
    return in.substr(first, last - first);
}

static std::vector<std::string> split_line(const std::string &line, char sep)
{
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= line.size()) {
        const size_t pos = line.find(sep, start);
        if (pos == std::string::npos) {
            parts.push_back(trim_copy(line.substr(start)));
            break;
        }
        parts.push_back(trim_copy(line.substr(start, pos - start)));
        start = pos + 1;
    }
    return parts;
}

static bool ends_with_ci(const std::string &value, const char *suffix)
{
    if (!suffix) {
        return false;
    }
    const size_t suffix_len = strlen(suffix);
    if (value.size() < suffix_len) {
        return false;
    }
    const size_t offset = value.size() - suffix_len;
    for (size_t i = 0; i < suffix_len; i++) {
        const unsigned char a = (unsigned char)value[offset + i];
        const unsigned char b = (unsigned char)suffix[i];
        if (tolower(a) != tolower(b)) {
            return false;
        }
    }
    return true;
}

static std::string basename_without_ext(const std::string &name)
{
    const size_t slash = name.find_last_of("/\\");
    const size_t start = (slash == std::string::npos) ? 0 : slash + 1;
    const size_t dot = name.find_last_of('.');
    if (dot != std::string::npos && dot > start) {
        return name.substr(start, dot - start);
    }
    return name.substr(start);
}

static std::string music_make_path(const std::string &rel)
{
    if (rel.empty()) {
        return "";
    }
    if (rel[0] == '/') {
        return rel;
    }
    std::string out = MUSIC_ROOT;
    if (out.back() != '/') {
        out.push_back('/');
    }
    out += rel;
    return out;
}

static std::string video_make_path(const char *base_dir, const std::string &rel)
{
    if (rel.empty()) {
        return "";
    }
    if (rel[0] == '/') {
        return rel;
    }
    std::string out = base_dir ? base_dir : VIDEO_ROOT;
    if (!out.empty() && out.back() != '/') {
        out.push_back('/');
    }
    out += rel;
    return out;
}

static bool video_load_manifest(const char *manifest_path,
                                const char *base_dir,
                                std::vector<video_item_t> *items)
{
    if (!items) {
        return false;
    }
    items->clear();
    FILE *fp = fopen(manifest_path, "r");
    if (!fp) {
        ESP_LOGW(TAG, "Video manifest missing: %s errno=%d", manifest_path, errno);
        return false;
    }

    char line[384];
    while (fgets(line, sizeof(line), fp)) {
        std::string raw = trim_copy(line);
        if (raw.empty() || raw[0] == '#') {
            continue;
        }
        std::vector<std::string> parts = split_line(raw, ',');
        if (parts.size() < 4) {
            parts = split_line(raw, '|');
        }
        if (parts.size() < 4 || parts[0] == "id") {
            continue;
        }
        video_item_t item = {};
        item.id = parts[0];
        item.title = parts[1];
        item.action = parts[2];
        item.path = video_make_path(base_dir, parts[3]);
        items->push_back(item);
    }
    fclose(fp);
    ESP_LOGI(TAG, "Loaded %u videos from %s", (unsigned)items->size(), manifest_path);
    return !items->empty();
}

static void video_reload_manifests(void)
{
    if (!s_sdcard_mounted) {
        s_course_videos.clear();
        s_offline_videos.clear();
        return;
    }
    video_load_manifest(COURSE_VIDEO_MANIFEST, COURSE_VIDEO_DIR, &s_course_videos);
    video_load_manifest(OFFLINE_VIDEO_MANIFEST, OFFLINE_VIDEO_DIR, &s_offline_videos);
}

static bool video_player_stop_current(void)
{
    if (!s_video_player_obj) {
        return true;
    }
    esp_lvgl_simple_player_stop();
    esp_err_t wait_ret = esp_lvgl_simple_player_wait_task_stop(3500);
    if (wait_ret != ESP_OK) {
        ESP_LOGE(TAG, "Video player did not stop cleanly, keep old instance");
        push_ui_msg("视频还在停止中，请稍后再试");
        return false;
    }
    esp_err_t del_ret = esp_lvgl_simple_player_del();
    if (del_ret != ESP_OK) {
        ESP_LOGE(TAG, "Video player delete failed: %s", esp_err_to_name(del_ret));
        return false;
    }
    s_video_player_obj = nullptr;
    s_video_player_path[0] = '\0';
    return true;
}

static void video_player_resize_current(lv_obj_t *host)
{
    if (!s_video_player_obj || !host) {
        return;
    }
    lv_obj_update_layout(host);
    int32_t w = lv_obj_get_width(host);
    int32_t h = lv_obj_get_height(host);
    if (w <= 0 || h <= 0) {
        w = lv_obj_get_content_width(host);
        h = lv_obj_get_content_height(host);
    }
    if (w <= 0 || h <= 0) {
        return;
    }
    lv_obj_set_pos(s_video_player_obj, 0, 0);
    lv_obj_set_size(s_video_player_obj, w, h);
    esp_lvgl_simple_player_resize((uint32_t)w, (uint32_t)h);
    lv_obj_invalidate(host);
}

static void video_player_pause_toggle_cb(lv_event_t *e)
{
    (void)e;
    if (!s_video_player_obj) {
        push_ui_msg("请先选择视频");
        return;
    }
    player_state_t state = esp_lvgl_simple_player_get_state();
    if (state == PLAYER_STATE_PLAYING) {
        esp_lvgl_simple_player_pause();
    } else if (state == PLAYER_STATE_PAUSED) {
        esp_lvgl_simple_player_play();
    }
}

static void video_player_stop_button_cb(lv_event_t *e)
{
    lv_obj_t *host = static_cast<lv_obj_t *>(lv_event_get_user_data(e));
    if (!host) {
        lv_obj_t *active = lv_scr_act();
        if (active == scr_courses) {
            host = s_course_player_host;
        } else if (active == scr_correction) {
            host = s_offline_player_host;
        }
    }
    if (video_player_stop_current() && host) {
        lv_obj_clean(host);
        make_text(host, "选择视频后播放预览", 106, 92, 190, font_14(), 0xd7dee8, LV_LABEL_LONG_DOT);
    }
}

static bool video_player_play_item(lv_obj_t *host, const video_item_t &item, lv_obj_t *status_label)
{
    if (!host) {
        return false;
    }
    if (!video_player_stop_current()) {
        if (status_label) {
            lv_label_set_text(status_label, "视频正在释放资源，请稍后重试。");
        }
        return false;
    }
    lv_obj_clean(host);
    if (item.path.empty() || access(item.path.c_str(), F_OK) != 0) {
        if (status_label) {
            lv_label_set_text(status_label, "未找到视频文件，请检查 SD 卡 FIT_VIDEO 目录。");
        }
        return false;
    }
    strlcpy(s_video_player_path, item.path.c_str(), sizeof(s_video_player_path));
    lv_obj_update_layout(host);
    int32_t host_w = lv_obj_get_width(host);
    int32_t host_h = lv_obj_get_height(host);
    if (host_w <= 0 || host_h <= 0) {
        host_w = lv_obj_get_content_width(host);
        host_h = lv_obj_get_content_height(host);
    }
    if (host_w <= 0) {
        host_w = 384;
    }
    if (host_h <= 0) {
        host_h = 216;
    }

    esp_lvgl_simple_player_cfg_t cfg = {};
    cfg.video_path = s_video_player_path;
    cfg.bgm_path = nullptr;
    cfg.screen = host;
    cfg.buff_size = 512 * 1024;
    cfg.cache_buff_size = 64 * 1024;
    cfg.cache_buff_in_psram = true;
    cfg.screen_width = (uint32_t)host_w;
    cfg.screen_height = (uint32_t)host_h;
    const bool sampled_video = item.path.find("/offline/") != std::string::npos ||
                               item.path.find("/results/") != std::string::npos;
    cfg.frame_delay_ms = sampled_video ? 167 : 100;
    cfg.flags.hide_controls = 1;
    cfg.flags.hide_slider = 1;
    cfg.flags.hide_status = 0;
    cfg.flags.auto_width = 0;
    cfg.flags.auto_height = 0;
    s_video_player_obj = esp_lvgl_simple_player_create(&cfg);
    if (!s_video_player_obj) {
        if (status_label) {
            lv_label_set_text(status_label, "播放器创建失败，请确认 MJPEG 格式和内存状态。");
        }
        return false;
    }
    lv_obj_set_pos(s_video_player_obj, 0, 0);
    lv_obj_set_size(s_video_player_obj, host_w, host_h);
    esp_lvgl_simple_player_repeat(true);
    esp_lvgl_simple_player_play();
    if (status_label) {
        lv_label_set_text(status_label, "正在播放 SD 卡演示视频。");
    }
    return true;
}

static std::string music_cover_for_id(const std::string &id)
{
    if (id.empty()) {
        return "";
    }
    const std::string cover_dir = std::string(MUSIC_ROOT) + "/covers";
    const char *exts[] = {".jpg", ".jpeg", ".JPG", ".JPEG"};
    for (const char *ext : exts) {
        std::string candidate = cover_dir + "/" + id + ext;
        if (access(candidate.c_str(), F_OK) == 0) {
            return candidate;
        }
    }

    DIR *dir = opendir(cover_dir.c_str());
    if (!dir) {
        return cover_dir + "/" + id + ".jpg";
    }
    std::string match;
    while (struct dirent *entry = readdir(dir)) {
        std::string name = entry->d_name;
        if (name.empty() || name[0] == '.') {
            continue;
        }
        if (!ends_with_ci(name, ".jpg") && !ends_with_ci(name, ".jpeg")) {
            continue;
        }
        if (basename_without_ext(name) == id) {
            match = cover_dir + "/" + name;
            break;
        }
    }
    closedir(dir);
    return match.empty() ? (cover_dir + "/" + id + ".jpg") : match;
}

static std::string music_title_from_id(const std::string &id)
{
    if (id == "huahai") {
        return "花海";
    }
    if (id == "liusha") {
        return "流沙";
    }
    return id.empty() ? "本地音乐" : id;
}

static void music_log_dir(const char *path)
{
    DIR *dir = opendir(path);
    if (!dir) {
        ESP_LOGW(TAG, "Music dir open failed: %s errno=%d", path, errno);
        return;
    }
    int count = 0;
    while (struct dirent *entry = readdir(dir)) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        ESP_LOGI(TAG, "Music dir %s entry: %s", path, entry->d_name);
        count++;
    }
    closedir(dir);
    ESP_LOGI(TAG, "Music dir %s entries=%d", path, count);
}

static size_t music_scan_tracks(std::vector<music_track_t> *tracks)
{
    if (!tracks) {
        return 0;
    }
    const std::string tracks_dir = std::string(MUSIC_ROOT) + "/tracks";
    DIR *dir = opendir(tracks_dir.c_str());
    if (!dir) {
        ESP_LOGW(TAG, "Music tracks dir missing: %s errno=%d", tracks_dir.c_str(), errno);
        return 0;
    }

    size_t added = 0;
    while (struct dirent *entry = readdir(dir)) {
        std::string name = entry->d_name;
        if (name.empty() || name[0] == '.') {
            continue;
        }
        if (!ends_with_ci(name, ".wav")) {
            continue;
        }
        const std::string id = basename_without_ext(name);
        music_track_t track = {};
        track.id = id;
        track.title = music_title_from_id(id);
        track.audio_path = tracks_dir + "/" + name;
        track.cover_path = music_cover_for_id(id);
        track.duration_ms = 0;
        tracks->push_back(track);
        added++;
        ESP_LOGI(TAG, "Music scan track: %s -> %s", track.id.c_str(), track.audio_path.c_str());
    }
    closedir(dir);
    std::sort(tracks->begin(), tracks->end(), [](const music_track_t &a, const music_track_t &b) {
        return a.id < b.id;
    });
    return added;
}

static void music_lock(void)
{
    if (s_music_mutex) {
        xSemaphoreTake(s_music_mutex, portMAX_DELAY);
    }
}

static void music_unlock(void)
{
    if (s_music_mutex) {
        xSemaphoreGive(s_music_mutex);
    }
}

static void music_set_status_locked(const char *status, bool error = false)
{
    strlcpy(s_music_state.status, status ? status : "", sizeof(s_music_state.status));
    s_music_state.error = error;
}

static music_state_t music_get_state_copy(void)
{
    music_state_t copy = {};
    music_lock();
    copy = s_music_state;
    music_unlock();
    return copy;
}

static void music_set_track_state_locked(const music_track_t *track, int index)
{
    s_music_state.index = index;
    if (track) {
        strlcpy(s_music_state.title, track->title.c_str(), sizeof(s_music_state.title));
        strlcpy(s_music_state.artist, "本地音乐", sizeof(s_music_state.artist));
        strlcpy(s_music_state.audio_path, track->audio_path.c_str(), sizeof(s_music_state.audio_path));
        strlcpy(s_music_state.cover_path, track->cover_path.c_str(), sizeof(s_music_state.cover_path));
        s_music_state.duration_ms = track->duration_ms;
    } else {
        strlcpy(s_music_state.title, "--", sizeof(s_music_state.title));
        strlcpy(s_music_state.artist, "本地音乐", sizeof(s_music_state.artist));
        s_music_state.audio_path[0] = '\0';
        s_music_state.cover_path[0] = '\0';
        s_music_state.duration_ms = 0;
    }
    s_music_state.position_ms = 0;
}

static bool music_load_playlist(void)
{
    std::vector<music_track_t> tracks;
    FILE *fp = fopen(MUSIC_PLAYLIST_PATH, "r");
    if (!fp) {
        ESP_LOGW(TAG, "Music playlist missing: %s errno=%d", MUSIC_PLAYLIST_PATH, errno);
        music_log_dir(MUSIC_ROOT);
    } else {
        ESP_LOGI(TAG, "Music playlist opened: %s", MUSIC_PLAYLIST_PATH);
        char line[384];
        while (fgets(line, sizeof(line), fp)) {
            std::string raw = trim_copy(line);
            if (raw.size() >= 3 &&
                (uint8_t)raw[0] == 0xef &&
                (uint8_t)raw[1] == 0xbb &&
                (uint8_t)raw[2] == 0xbf) {
                raw.erase(0, 3);
            }
            if (raw.empty() || raw[0] == '#') {
                continue;
            }
            std::vector<std::string> parts = split_line(raw, '|');
            if (parts.size() < 4) {
                parts = split_line(raw, ',');
            }
            if (parts.size() < 4 || parts[1].empty() || parts[2].empty()) {
                ESP_LOGW(TAG, "Skip invalid playlist row: %s", raw.c_str());
                continue;
            }
            music_track_t track = {};
            track.id = parts[0].empty() ? basename_without_ext(parts[2]) : parts[0];
            track.title = parts[1];
            track.audio_path = music_make_path(parts[2]);
            track.cover_path = parts[3].empty() ? music_cover_for_id(track.id) : music_make_path(parts[3]);
            track.duration_ms = 0;
            tracks.push_back(track);
            ESP_LOGI(TAG, "Music playlist track: %s audio=%s cover=%s",
                     track.title.c_str(),
                     track.audio_path.c_str(),
                     track.cover_path.c_str());
        }
        fclose(fp);
    }

    if (tracks.empty()) {
        ESP_LOGW(TAG, "Music playlist empty, fallback scan tracks dir");
        music_scan_tracks(&tracks);
    }

    music_lock();
    s_music_tracks = tracks;
    s_music_state.sd_ready = s_sdcard_mounted;
    s_music_state.library_ready = !s_music_tracks.empty();
    if (!s_music_tracks.empty()) {
        const int index = (s_music_state.index >= 0 && s_music_state.index < (int)s_music_tracks.size())
                              ? s_music_state.index
                              : 0;
        music_set_track_state_locked(&s_music_tracks[index], index);
        music_set_status_locked("曲库就绪");
    } else {
        music_set_track_state_locked(nullptr, -1);
        music_set_status_locked("曲库为空", true);
    }
    music_unlock();
    ESP_LOGI(TAG, "Music playlist loaded: %u tracks", (unsigned)tracks.size());
    return !tracks.empty();
}

static uint16_t read_le16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_le32(const uint8_t *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

typedef struct {
    uint16_t format;
    uint16_t channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    uint32_t data_offset;
    uint32_t data_size;
    uint32_t duration_ms;
} wav_info_t;

static bool wav_parse(FILE *fp, wav_info_t *info)
{
    if (!fp || !info) {
        return false;
    }
    memset(info, 0, sizeof(*info));
    uint8_t header[12];
    if (fread(header, 1, sizeof(header), fp) != sizeof(header)) {
        return false;
    }
    if (memcmp(header, "RIFF", 4) != 0 || memcmp(header + 8, "WAVE", 4) != 0) {
        return false;
    }

    bool have_fmt = false;
    bool have_data = false;
    while (!have_data) {
        uint8_t chunk[8];
        if (fread(chunk, 1, sizeof(chunk), fp) != sizeof(chunk)) {
            break;
        }
        const uint32_t size = read_le32(chunk + 4);
        const long chunk_data_pos = ftell(fp);
        if (memcmp(chunk, "fmt ", 4) == 0) {
            uint8_t fmt[40] = {};
            const uint32_t to_read = size < sizeof(fmt) ? size : sizeof(fmt);
            if (fread(fmt, 1, to_read, fp) != to_read) {
                return false;
            }
            info->format = read_le16(fmt + 0);
            info->channels = read_le16(fmt + 2);
            info->sample_rate = read_le32(fmt + 4);
            info->byte_rate = read_le32(fmt + 8);
            info->block_align = read_le16(fmt + 12);
            info->bits_per_sample = read_le16(fmt + 14);
            have_fmt = true;
        } else if (memcmp(chunk, "data", 4) == 0) {
            info->data_offset = (uint32_t)ftell(fp);
            info->data_size = size;
            have_data = true;
        }

        const long next = chunk_data_pos + (long)size + (size & 1);
        if (fseek(fp, next, SEEK_SET) != 0) {
            break;
        }
    }

    if (!have_fmt || !have_data || info->byte_rate == 0) {
        return false;
    }
    info->duration_ms = (uint32_t)(((uint64_t)info->data_size * 1000ULL) / info->byte_rate);
    return true;
}

static bool music_ensure_audio(void)
{
    if (s_music_audio_ready && s_music_codec) {
        return true;
    }
    s_music_codec = bsp_audio_codec_speaker_init();
    if (!s_music_codec) {
        ESP_LOGE(TAG, "Music speaker codec init failed");
        return false;
    }
    esp_codec_dev_set_out_vol(s_music_codec, MUSIC_DEFAULT_VOLUME);
    s_music_audio_ready = true;
    return true;
}

static void music_set_volume(int volume)
{
    if (volume < 0) {
        volume = 0;
    }
    if (volume > 100) {
        volume = 100;
    }
    music_lock();
    s_music_state.volume = volume;
    music_unlock();
    if (s_music_codec) {
        esp_codec_dev_set_out_vol(s_music_codec, volume);
    }
}

static void music_send_cmd(MusicCmdType type, int value = 0)
{
    if (!s_music_cmd_q) {
        return;
    }
    music_cmd_t cmd = {.type = type, .value = value};
    xQueueSend(s_music_cmd_q, &cmd, 0);
}

static bool training_voice_fetch_pcm(VoicePromptId prompt, voice_pcm_t *out);
static void training_voice_release_pcm(voice_pcm_t *audio);

static bool music_get_track_copy(int index, music_track_t *out)
{
    if (!out) {
        return false;
    }
    music_lock();
    const bool ok = index >= 0 && index < (int)s_music_tracks.size();
    if (ok) {
        *out = s_music_tracks[index];
    }
    music_unlock();
    return ok;
}

static int music_track_count(void)
{
    music_lock();
    const int count = (int)s_music_tracks.size();
    music_unlock();
    return count;
}

static int music_normalize_index(int index)
{
    const int count = music_track_count();
    if (count <= 0) {
        return -1;
    }
    while (index < 0) {
        index += count;
    }
    return index % count;
}

static void music_playback_task(void *arg)
{
    (void)arg;
    uint8_t *buf = (uint8_t *)heap_caps_malloc(MUSIC_IO_BUFFER_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) {
        buf = (uint8_t *)heap_caps_malloc(MUSIC_IO_BUFFER_BYTES, MALLOC_CAP_8BIT);
    }
    if (!buf) {
        music_lock();
        music_set_status_locked("音乐缓冲分配失败", true);
        music_unlock();
        vTaskDelete(nullptr);
        return;
    }

    int current_index = -1;
    FILE *fp = nullptr;
    wav_info_t wav = {};
    uint32_t remaining = 0;
    uint32_t played_bytes = 0;
    bool paused = false;

    auto close_current = [&]() {
        if (fp) {
            fclose(fp);
            fp = nullptr;
        }
        if (s_music_codec) {
            esp_codec_dev_close(s_music_codec);
        }
        remaining = 0;
        played_bytes = 0;
        paused = false;
        music_lock();
        s_music_state.playing = false;
        s_music_state.paused = false;
        s_music_state.position_ms = 0;
        music_unlock();
    };

    auto open_index = [&](int index) -> bool {
        close_current();
        index = music_normalize_index(index);
        if (index < 0) {
            music_lock();
            music_set_status_locked("没有可播放曲目", true);
            music_unlock();
            return false;
        }
        music_track_t track;
        if (!music_get_track_copy(index, &track)) {
            return false;
        }
        if (!music_ensure_audio()) {
            music_lock();
            music_set_status_locked("音频初始化失败", true);
            music_unlock();
            return false;
        }

        fp = fopen(track.audio_path.c_str(), "rb");
        if (!fp) {
            ESP_LOGE(TAG, "Open music failed: %s errno=%d", track.audio_path.c_str(), errno);
            music_lock();
            music_set_track_state_locked(&track, index);
            music_set_status_locked("打开音乐失败", true);
            music_unlock();
            return false;
        }
        if (!wav_parse(fp, &wav) ||
            wav.format != 1 ||
            wav.bits_per_sample != 16 ||
            wav.channels < 1 ||
            wav.channels > 2 ||
            wav.data_size == 0) {
            ESP_LOGE(TAG,
                     "Unsupported WAV: fmt=%u bits=%u ch=%u path=%s",
                     wav.format,
                     wav.bits_per_sample,
                     wav.channels,
                     track.audio_path.c_str());
            fclose(fp);
            fp = nullptr;
            music_lock();
            music_set_track_state_locked(&track, index);
            music_set_status_locked("仅支持16bit PCM WAV", true);
            music_unlock();
            return false;
        }
        if (fseek(fp, wav.data_offset, SEEK_SET) != 0) {
            fclose(fp);
            fp = nullptr;
            music_lock();
            music_set_status_locked("WAV定位失败", true);
            music_unlock();
            return false;
        }

        esp_codec_dev_sample_info_t fs = {
            .bits_per_sample = (uint8_t)wav.bits_per_sample,
            .channel = (uint8_t)wav.channels,
            .channel_mask = 0,
            .sample_rate = wav.sample_rate,
            .mclk_multiple = 256,
        };
        if (esp_codec_dev_open(s_music_codec, &fs) != 0) {
            fclose(fp);
            fp = nullptr;
            music_lock();
            music_set_status_locked("Codec打开失败", true);
            music_unlock();
            return false;
        }

        current_index = index;
        remaining = wav.data_size;
        played_bytes = 0;
        paused = false;
        track.duration_ms = wav.duration_ms;
        music_lock();
        if (index >= 0 && index < (int)s_music_tracks.size()) {
            s_music_tracks[index].duration_ms = wav.duration_ms;
        }
        music_set_track_state_locked(&track, index);
        s_music_state.duration_ms = wav.duration_ms;
        s_music_state.playing = true;
        s_music_state.paused = false;
        s_music_state.position_ms = 0;
        music_set_status_locked("正在播放");
        music_unlock();
        ESP_LOGI(TAG,
                 "Music play: %s %uHz %uch %ubit duration=%ums",
                 track.title.c_str(),
                 (unsigned)wav.sample_rate,
                 (unsigned)wav.channels,
                 (unsigned)wav.bits_per_sample,
                 (unsigned)wav.duration_ms);
        return true;
    };

    auto reopen_current_codec = [&]() -> bool {
        if (!fp || !s_music_codec) {
            return false;
        }
        esp_codec_dev_sample_info_t fs = {
            .bits_per_sample = (uint8_t)wav.bits_per_sample,
            .channel = (uint8_t)wav.channels,
            .channel_mask = 0,
            .sample_rate = wav.sample_rate,
            .mclk_multiple = 256,
        };
        if (esp_codec_dev_open(s_music_codec, &fs) != 0) {
            ESP_LOGW(TAG, "Music codec reopen after voice failed");
            close_current();
            music_lock();
            music_set_status_locked("音乐恢复失败", true);
            music_unlock();
            return false;
        }
        return true;
    };

    auto play_voice_prompt = [&](VoicePromptId prompt) {
        music_lock();
        const bool had_track = fp != nullptr;
        const bool was_paused = paused;
        music_set_status_locked("训练语音播报");
        music_unlock();

        voice_pcm_t voice = {};
        if (!training_voice_fetch_pcm(prompt, &voice)) {
            music_lock();
            music_set_status_locked(had_track ? (was_paused ? "已暂停" : "正在播放") : "语音获取失败", !had_track);
            music_unlock();
            return;
        }

        if (!music_ensure_audio()) {
            training_voice_release_pcm(&voice);
            music_lock();
            music_set_status_locked("音频初始化失败", true);
            music_unlock();
            return;
        }

        if (s_music_codec) {
            esp_codec_dev_close(s_music_codec);
        }
        esp_codec_dev_set_out_vol(s_music_codec, MUSIC_DEFAULT_VOLUME);

        esp_codec_dev_sample_info_t fs = {
            .bits_per_sample = (uint8_t)voice.bits_per_sample,
            .channel = (uint8_t)voice.channels,
            .channel_mask = 0,
            .sample_rate = voice.sample_rate,
            .mclk_multiple = 256,
        };

        bool played_voice = false;
        if (esp_codec_dev_open(s_music_codec, &fs) == 0) {
            size_t off = 0;
            while (off < voice.len) {
                const size_t chunk = std::min<size_t>(MUSIC_IO_BUFFER_BYTES, voice.len - off);
                if (esp_codec_dev_write(s_music_codec, voice.data + off, (int)chunk) != 0) {
                    ESP_LOGW(TAG, "Voice write failed at %u/%u", (unsigned)off, (unsigned)voice.len);
                    break;
                }
                off += chunk;
            }
            played_voice = off >= voice.len;
            esp_codec_dev_close(s_music_codec);
        } else {
            ESP_LOGW(TAG, "Voice codec open failed");
        }
        training_voice_release_pcm(&voice);

        if (had_track && fp) {
            if (reopen_current_codec()) {
                paused = was_paused;
                music_lock();
                s_music_state.playing = true;
                s_music_state.paused = paused;
                music_set_status_locked(paused ? "已暂停" : "正在播放");
                music_unlock();
            }
        } else {
            music_lock();
            s_music_state.playing = false;
            s_music_state.paused = false;
            music_set_status_locked(played_voice ? "语音播报完成" : "语音播报失败", !played_voice);
            music_unlock();
        }
    };

    auto handle_cmd = [&](const music_cmd_t &cmd) {
        if (cmd.type == MusicCmdType::PlayIndex) {
            open_index(cmd.value);
        } else if (cmd.type == MusicCmdType::Toggle) {
            if (!fp) {
                music_state_t st = music_get_state_copy();
                open_index(st.index >= 0 ? st.index : 0);
            } else {
                paused = !paused;
                music_lock();
                s_music_state.paused = paused;
                music_set_status_locked(paused ? "已暂停" : "正在播放");
                music_unlock();
            }
        } else if (cmd.type == MusicCmdType::Stop) {
            close_current();
            music_lock();
            music_set_status_locked("已停止");
            music_unlock();
        } else if (cmd.type == MusicCmdType::Next) {
            open_index(music_normalize_index(current_index + 1));
        } else if (cmd.type == MusicCmdType::Prev) {
            open_index(music_normalize_index(current_index - 1));
        } else if (cmd.type == MusicCmdType::SetVolume) {
            music_set_volume(cmd.value);
        } else if (cmd.type == MusicCmdType::Rescan) {
            close_current();
            music_load_playlist();
        } else if (cmd.type == MusicCmdType::VoicePrompt) {
            play_voice_prompt((VoicePromptId)cmd.value);
        }
    };

    while (true) {
        music_cmd_t cmd = {};
        if (!fp) {
            if (xQueueReceive(s_music_cmd_q, &cmd, portMAX_DELAY) == pdTRUE) {
                handle_cmd(cmd);
            }
            continue;
        }

        while (xQueueReceive(s_music_cmd_q, &cmd, 0) == pdTRUE) {
            handle_cmd(cmd);
        }
        if (!fp) {
            continue;
        }
        if (paused) {
            if (xQueueReceive(s_music_cmd_q, &cmd, pdMS_TO_TICKS(80)) == pdTRUE) {
                handle_cmd(cmd);
            }
            continue;
        }

        const size_t ask = remaining > MUSIC_IO_BUFFER_BYTES ? MUSIC_IO_BUFFER_BYTES : remaining;
        if (ask == 0) {
            open_index(music_normalize_index(current_index + 1));
            continue;
        }
        const size_t got = fread(buf, 1, ask, fp);
        if (got == 0) {
            open_index(music_normalize_index(current_index + 1));
            continue;
        }
        const int ret = esp_codec_dev_write(s_music_codec, buf, (int)got);
        if (ret != 0) {
            ESP_LOGW(TAG, "Music write failed ret=%d", ret);
            close_current();
            music_lock();
            music_set_status_locked("音频写入失败", true);
            music_unlock();
            continue;
        }
        remaining -= (uint32_t)got;
        played_bytes += (uint32_t)got;
        music_lock();
        s_music_state.position_ms = wav.byte_rate ? (uint32_t)(((uint64_t)played_bytes * 1000ULL) / wav.byte_rate) : 0;
        s_music_state.playing = true;
        s_music_state.paused = false;
        music_unlock();
    }
}

static void music_start_service(void)
{
    if (!s_music_mutex) {
        s_music_mutex = xSemaphoreCreateMutex();
    }
    if (!s_music_cmd_q) {
        s_music_cmd_q = xQueueCreate(8, sizeof(music_cmd_t));
    }
    music_lock();
    s_music_state.sd_ready = s_sdcard_mounted;
    music_unlock();
    if (s_sdcard_mounted) {
        music_load_playlist();
    }
    if (!s_music_task && s_music_cmd_q) {
        xTaskCreatePinnedToCore(music_playback_task, "music_player", 8192, nullptr, 4, &s_music_task, 0);
    }
}

static void sdcard_retry_task(void *arg)
{
    (void)arg;
    const uint32_t delays_ms[] = {1200, 1800, 2500, 3500, 5000, 8000};
    for (size_t i = 0; i < sizeof(delays_ms) / sizeof(delays_ms[0]); i++) {
        vTaskDelay(pdMS_TO_TICKS(delays_ms[i]));
        if (s_sdcard_mounted) {
            break;
        }
        ESP_LOGI(TAG, "TF/SD card retry mount %u/%u",
                 (unsigned)(i + 1),
                 (unsigned)(sizeof(delays_ms) / sizeof(delays_ms[0])));
        esp_err_t err = bsp_sdcard_mount();
        if (err == ESP_OK) {
            s_sdcard_mounted = true;
            ESP_LOGI(TAG, "TF/SD card mounted by retry");
            music_lock();
            s_music_state.sd_ready = true;
            music_set_status_locked("TF卡已挂载");
            music_unlock();
            music_load_playlist();
            video_reload_manifests();
            break;
        }
        ESP_LOGW(TAG, "TF/SD retry mount failed: %s", esp_err_to_name(err));
    }
    s_sdcard_retry_task = nullptr;
    vTaskDelete(nullptr);
}

static void start_sdcard_retry_task(void)
{
    if (!s_sdcard_retry_task) {
        xTaskCreatePinnedToCore(sdcard_retry_task, "sdcard_retry", 4096, nullptr, 3, &s_sdcard_retry_task, 0);
    }
}

static void set_label(lv_obj_t *label, const char *text)
{
    if (!label) {
        mark_lvgl_build_failed("set label");
        return;
    }
    lv_label_set_text(label, text ? text : "");
    lv_obj_set_style_text_font(label, font_18(), LV_PART_MAIN);
    lv_obj_set_style_text_letter_space(label, 0, LV_PART_MAIN);
}

static void style_screen(lv_obj_t *scr)
{
    if (!scr) {
        mark_lvgl_build_failed("screen");
        return;
    }
    lv_obj_set_size(scr, SCREEN_W, SCREEN_H);
    lv_obj_clear_flag(scr, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_color(scr, lv_color_hex(0xf4f6fb), LV_PART_MAIN);
    lv_obj_set_style_pad_all(scr, 0, LV_PART_MAIN);
}

static lv_obj_t *make_button(lv_obj_t *parent, int32_t w, int32_t h, const char *text)
{
    if (!parent) {
        mark_lvgl_build_failed("button parent");
        return nullptr;
    }
    lv_obj_t *btn = lv_btn_create(parent);
    if (!btn) {
        mark_lvgl_build_failed("button");
        return nullptr;
    }
    lv_obj_set_size(btn, w, h);
    lv_obj_set_style_radius(btn, 8, LV_PART_MAIN);

    lv_obj_t *label = lv_label_create(btn);
    if (!label) {
        mark_lvgl_build_failed("button label");
        return btn;
    }
    set_label(label, text);
    lv_obj_center(label);
    return btn;
}

static lv_obj_t *make_preview(lv_obj_t *parent)
{
    if (!parent) {
        mark_lvgl_build_failed("preview parent");
        return nullptr;
    }
    lv_obj_t *view = lv_obj_create(parent);
    if (!view) {
        mark_lvgl_build_failed("preview");
        return nullptr;
    }
    lv_obj_set_size(view, PREVIEW_W, PREVIEW_H);
    lv_obj_align(view, LV_ALIGN_LEFT_MID, 0, 0);
    lv_obj_clear_flag(view, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_clear_flag(view, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_set_style_bg_color(view, lv_color_black(), LV_PART_MAIN);
    lv_obj_set_style_border_width(view, 0, LV_PART_MAIN);
    lv_obj_set_style_radius(view, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_all(view, 0, LV_PART_MAIN);
    return view;
}

static void place_canvas(lv_obj_t *canvas, lv_obj_t *view, uint32_t w, uint32_t h)
{
    if (!canvas || !view || w == 0 || h == 0) {
        return;
    }

    lv_obj_set_size(canvas, (int32_t)w, (int32_t)h);
    lv_obj_center(canvas);
}

static lv_obj_t *make_side_panel(lv_obj_t *parent)
{
    if (!parent) {
        mark_lvgl_build_failed("side panel parent");
        return nullptr;
    }
    lv_obj_t *panel = lv_obj_create(parent);
    if (!panel) {
        mark_lvgl_build_failed("side panel");
        return nullptr;
    }
    lv_obj_set_size(panel, SIDE_PANEL_W, SCREEN_H);
    lv_obj_set_pos(panel, SIDE_PANEL_X, 0);
    lv_obj_clear_flag(panel, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_color(panel, lv_color_hex(0xf7f9fc), LV_PART_MAIN);
    lv_obj_set_style_border_width(panel, 0, LV_PART_MAIN);
    lv_obj_set_style_radius(panel, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_all(panel, 12, LV_PART_MAIN);
    return panel;
}

static void show_menu_cb(lv_event_t *e);
static void show_start_cb(lv_event_t *e);
static void show_correction_cb(lv_event_t *e);
static void show_courses_cb(lv_event_t *e);
static void show_plan_cb(lv_event_t *e);
static void show_report_cb(lv_event_t *e);
static void show_settings_cb(lv_event_t *e);
static void show_music_cb(lv_event_t *e);
static void show_pose_cb(lv_event_t *e);
static void profile_choice_cb(lv_event_t *e);
static void profile_save_cb(lv_event_t *e);
static void profile_default_cb(lv_event_t *e);
static void profile_edit_cb(lv_event_t *e);
static void training_card_cb(lv_event_t *e);
static void training_start_selected_cb(lv_event_t *e);
static void training_goal_adjust_cb(lv_event_t *e);
static void plan_toggle_week_cb(lv_event_t *e);
static void training_toggle_cb(lv_event_t *e);
static void training_finish_cb(lv_event_t *e);
static void music_play_toggle_cb(lv_event_t *e);
static void music_stop_cb(lv_event_t *e);
static void music_next_cb(lv_event_t *e);
static void course_video_select_cb(lv_event_t *e);
static void course_video_refresh_cb(lv_event_t *e);
static void course_video_zoom_cb(lv_event_t *e);
static void video_player_pause_toggle_cb(lv_event_t *e);
static void video_player_stop_button_cb(lv_event_t *e);
static void offline_video_select_cb(lv_event_t *e);
static void offline_video_process_cb(lv_event_t *e);
static void offline_video_delete_result_cb(lv_event_t *e);
static void offline_video_refresh_cb(lv_event_t *e);
static void offline_video_zoom_cb(lv_event_t *e);
static void offline_video_process_task(void *arg);
static void offline_render_list(void);
static void offline_update_selected_status(void);
static void music_prev_cb(lv_event_t *e);
static void music_track_cb(lv_event_t *e);
static void music_volume_cb(lv_event_t *e);
static void music_rescan_cb(lv_event_t *e);

static void create_menu(void);
static void create_camera_page(void);
static void create_pose_page(void);
static void create_start_page(void);
static void create_correction_page(void);
static void create_courses_page(void);
static void create_plan_page(void);
static void create_report_page(void);
static void create_settings_page(void);
static void create_music_page(void);
static void create_face_page(void);
static void create_auth_page(void);
static void create_profile_page(void);
static bool pc_pose_wifi_start(void);

typedef void (*create_product_page_fn_t)(void);
static void page_preload_timer_cb(lv_timer_t *timer);
static void start_page_preloader(void);
static bool create_page_checked(lv_obj_t **screen, create_product_page_fn_t create_fn, const char *name);
static void destroy_page_if_inactive(lv_obj_t **screen);
static void profile_sync_edit_state(void);
static void update_music_ui(void);
static void music_send_cmd(MusicCmdType type, int value);
static void music_start_service(void);

static lv_timer_t *s_page_preload_timer = nullptr;
static int s_page_preload_index = 0;
static bool s_page_preload_complete = false;
static lv_obj_t **s_pending_screen = nullptr;
static create_product_page_fn_t s_pending_create_fn = nullptr;
static const char *s_pending_page_name = nullptr;
static bool s_pending_start_pose = false;
static lv_obj_t *s_boot_logo = nullptr;
static lv_obj_t *s_boot_bar = nullptr;
static lv_obj_t *s_boot_percent = nullptr;
static lv_obj_t *s_boot_stage = nullptr;
static lv_obj_t *s_boot_log_labels[7] = {};
static int s_boot_log_count = 0;

static const lv_font_t *font_14(void)
{
    return &font_zh_14;
}

static const lv_font_t *font_16(void)
{
    return &font_zh_16;
}

static const lv_font_t *font_18(void)
{
    return &font_zh_18;
}

static void style_product_screen(lv_obj_t *scr)
{
    if (!scr) {
        mark_lvgl_build_failed("product screen");
        return;
    }
    lv_obj_set_size(scr, SCREEN_W, SCREEN_H);
    lv_obj_clear_flag(scr, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_color(scr, lv_color_hex(0xf4f7f8), LV_PART_MAIN);
    lv_obj_set_style_pad_all(scr, 0, LV_PART_MAIN);
}

static void style_text(lv_obj_t *obj, const lv_font_t *font, uint32_t color)
{
    if (!obj) {
        return;
    }
    lv_obj_set_style_text_font(obj, font, LV_PART_MAIN);
    lv_obj_set_style_text_color(obj, lv_color_hex(color), LV_PART_MAIN);
    lv_obj_set_style_text_letter_space(obj, 0, LV_PART_MAIN);
    lv_obj_set_style_text_line_space(obj, 4, LV_PART_MAIN);
}

static lv_obj_t *make_text(lv_obj_t *parent,
                           const char *text,
                           int32_t x,
                           int32_t y,
                           int32_t w,
                           const lv_font_t *font,
                           uint32_t color,
                           lv_label_long_mode_t mode = LV_LABEL_LONG_WRAP)
{
    if (!parent) {
        mark_lvgl_build_failed("label parent");
        return nullptr;
    }
    lv_obj_t *label = lv_label_create(parent);
    if (!label) {
        mark_lvgl_build_failed("label");
        return nullptr;
    }
    lv_label_set_text(label, text ? text : "");
    lv_obj_set_pos(label, x, y);
    lv_obj_set_width(label, w);
    lv_label_set_long_mode(label, mode);
    style_text(label, font, color);
    return label;
}

static lv_obj_t *make_card(lv_obj_t *parent, int32_t x, int32_t y, int32_t w, int32_t h)
{
    if (!parent) {
        mark_lvgl_build_failed("card parent");
        return nullptr;
    }
    lv_obj_t *card = lv_obj_create(parent);
    if (!card) {
        mark_lvgl_build_failed("card");
        return nullptr;
    }
    lv_obj_set_pos(card, x, y);
    lv_obj_set_size(card, w, h);
    lv_obj_clear_flag(card, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_color(card, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_border_color(card, lv_color_hex(0xd9e2e8), LV_PART_MAIN);
    lv_obj_set_style_border_width(card, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(card, 8, LV_PART_MAIN);
    lv_obj_set_style_pad_all(card, 14, LV_PART_MAIN);
    lv_obj_set_style_shadow_width(card, 10, LV_PART_MAIN);
    lv_obj_set_style_shadow_opa(card, LV_OPA_10, LV_PART_MAIN);
    lv_obj_set_style_shadow_color(card, lv_color_hex(0x102330), LV_PART_MAIN);
    return card;
}

static lv_obj_t *make_dot(lv_obj_t *parent, int32_t x, int32_t y, uint32_t color)
{
    if (!parent) {
        mark_lvgl_build_failed("dot parent");
        return nullptr;
    }
    lv_obj_t *dot = lv_obj_create(parent);
    if (!dot) {
        mark_lvgl_build_failed("dot");
        return nullptr;
    }
    lv_obj_remove_style_all(dot);
    lv_obj_set_pos(dot, x, y);
    lv_obj_set_size(dot, 8, 8);
    lv_obj_set_style_radius(dot, 8, LV_PART_MAIN);
    lv_obj_set_style_bg_color(dot, lv_color_hex(color), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(dot, LV_OPA_COVER, LV_PART_MAIN);
    return dot;
}

static lv_obj_t *make_pill(lv_obj_t *parent, int32_t x, int32_t y, int32_t w, const char *text, uint32_t dot_color)
{
    if (!parent) {
        mark_lvgl_build_failed("pill parent");
        return nullptr;
    }
    lv_obj_t *pill = lv_obj_create(parent);
    if (!pill) {
        mark_lvgl_build_failed("pill");
        return nullptr;
    }
    lv_obj_set_pos(pill, x, y);
    lv_obj_set_size(pill, w, 32);
    lv_obj_clear_flag(pill, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_color(pill, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_border_color(pill, lv_color_hex(0xd9e2e8), LV_PART_MAIN);
    lv_obj_set_style_border_width(pill, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(pill, 16, LV_PART_MAIN);
    lv_obj_set_style_pad_all(pill, 0, LV_PART_MAIN);
    make_dot(pill, 12, 12, dot_color);
    make_text(pill, text, 28, 7, w - 36, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    return pill;
}

static void make_progress(lv_obj_t *parent, int32_t x, int32_t y, int32_t w, int32_t percent, uint32_t color)
{
    if (!parent) {
        mark_lvgl_build_failed("progress parent");
        return;
    }
    lv_obj_t *track = lv_obj_create(parent);
    if (!track) {
        mark_lvgl_build_failed("progress track");
        return;
    }
    lv_obj_remove_style_all(track);
    lv_obj_set_pos(track, x, y);
    lv_obj_set_size(track, w, 8);
    lv_obj_set_style_radius(track, 8, LV_PART_MAIN);
    lv_obj_set_style_bg_color(track, lv_color_hex(0xe4edf1), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(track, LV_OPA_COVER, LV_PART_MAIN);

    lv_obj_t *bar = lv_obj_create(track);
    if (!bar) {
        mark_lvgl_build_failed("progress bar");
        return;
    }
    lv_obj_remove_style_all(bar);
    lv_obj_set_pos(bar, 0, 0);
    lv_obj_set_size(bar, (w * percent) / 100, 8);
    lv_obj_set_style_radius(bar, 8, LV_PART_MAIN);
    lv_obj_set_style_bg_color(bar, lv_color_hex(color), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(bar, LV_OPA_COVER, LV_PART_MAIN);
}

static lv_obj_t *make_product_button(lv_obj_t *parent,
                                     int32_t x,
                                     int32_t y,
                                     int32_t w,
                                     int32_t h,
                                     const char *text,
                                     lv_event_cb_t cb,
                                     bool primary,
                                     void *user_data = nullptr)
{
    lv_obj_t *btn = lv_btn_create(parent);
    lv_obj_set_pos(btn, x, y);
    lv_obj_set_size(btn, w, h);
    lv_obj_set_style_radius(btn, 8, LV_PART_MAIN);
    lv_obj_set_style_border_width(btn, primary ? 0 : 1, LV_PART_MAIN);
    lv_obj_set_style_border_color(btn, lv_color_hex(0xd9e2e8), LV_PART_MAIN);
    lv_obj_set_style_bg_color(btn, lv_color_hex(primary ? 0x145883 : 0xffffff), LV_PART_MAIN);
    lv_obj_set_style_shadow_width(btn, 0, LV_PART_MAIN);
    if (cb) {
        lv_obj_add_event_cb(btn, cb, LV_EVENT_CLICKED, user_data);
    }

    lv_obj_t *label = lv_label_create(btn);
    if (!label) {
        mark_lvgl_build_failed("button label");
        return btn;
    }
    lv_label_set_text(label, text ? text : "");
    lv_obj_set_width(label, std::max<int32_t>(8, w - 14));
    lv_label_set_long_mode(label, LV_LABEL_LONG_DOT);
    lv_obj_set_style_text_align(label, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
    style_text(label, font_14(), primary ? 0xffffff : 0x314551);
    lv_obj_center(label);
    return btn;
}

static void set_button_text(lv_obj_t *btn, const char *text)
{
    if (!btn) {
        return;
    }
    lv_obj_t *label = lv_obj_get_child(btn, 0);
    if (label) {
        lv_label_set_text(label, text ? text : "");
        lv_obj_center(label);
    }
}

static lv_obj_t *make_profile_choice_button(lv_obj_t *parent,
                                            int32_t x,
                                            int32_t y,
                                            int32_t w,
                                            const char *text,
                                            const char *data)
{
    lv_obj_t *btn = lv_btn_create(parent);
    if (!btn) {
        mark_lvgl_build_failed("profile choice");
        return nullptr;
    }
    lv_obj_set_pos(btn, x, y);
    lv_obj_set_size(btn, w, 34);
    lv_obj_set_style_radius(btn, 8, LV_PART_MAIN);
    lv_obj_set_style_shadow_width(btn, 0, LV_PART_MAIN);
    lv_obj_set_style_border_width(btn, 1, LV_PART_MAIN);
    lv_obj_set_style_border_color(btn, lv_color_hex(0xcbd8df), LV_PART_MAIN);
    lv_obj_set_style_bg_color(btn, lv_color_white(), LV_PART_MAIN);
    lv_obj_add_event_cb(btn, profile_choice_cb, LV_EVENT_CLICKED, (void *)data);

    lv_obj_t *label = lv_label_create(btn);
    if (!label) {
        mark_lvgl_build_failed("profile choice label");
        return btn;
    }
    lv_label_set_text(label, text ? text : "");
    style_text(label, font_14(), 0x314551);
    lv_obj_center(label);
    return btn;
}

static void make_metric(lv_obj_t *parent,
                        int32_t x,
                        int32_t y,
                        int32_t w,
                        int32_t h,
                        const char *label,
                        const char *value,
                        int32_t percent,
                        uint32_t color)
{
    lv_obj_t *card = make_card(parent, x, y, w, h);
    if (!card) {
        return;
    }
    make_text(card, label, 0, 0, w - 28, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_text(card, value, 0, 28, w - 28, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    if (percent >= 0) {
        make_progress(card, 0, h - 28, w - 28, percent, color);
    }
}

struct ProductNavItem {
    const char *id;
    const char *icon;
    const char *text;
    lv_event_cb_t cb;
};

static const ProductNavItem PRODUCT_NAV[] = {
    {"overview", "H", "首页", show_menu_cb},
    {"start", "T", "选择训练", show_start_cb},
    {"live", "L", "实时训练", show_pose_cb},
    {"correction", "C", "离线处理", show_correction_cb},
    {"courses", "B", "课程库", show_courses_cb},
    {"plan", "P", "训练计划", show_plan_cb},
    {"music", "M", "音乐", show_music_cb},
    {"report", "R", "训练报告", show_report_cb},
    {"settings", "S", "设备设置", show_settings_cb},
};

static lv_obj_t *make_nav_button(lv_obj_t *parent, int32_t y, const ProductNavItem &item, const char *active_id)
{
    if (!parent) {
        mark_lvgl_build_failed("nav button parent");
        return nullptr;
    }
    const bool active = strcmp(item.id, active_id) == 0;
    lv_obj_t *btn = lv_btn_create(parent);
    if (!btn) {
        mark_lvgl_build_failed("nav button");
        return nullptr;
    }
    lv_obj_set_pos(btn, 14, y);
    lv_obj_set_size(btn, 188, 38);
    lv_obj_set_style_radius(btn, 8, LV_PART_MAIN);
    lv_obj_set_style_shadow_width(btn, 0, LV_PART_MAIN);
    lv_obj_set_style_bg_color(btn, lv_color_hex(active ? 0xe8f2f6 : 0xffffff), LV_PART_MAIN);
    lv_obj_set_style_border_width(btn, 0, LV_PART_MAIN);
    lv_obj_add_event_cb(btn, item.cb, LV_EVENT_CLICKED, NULL);

    lv_obj_t *mark = lv_obj_create(btn);
    if (!mark) {
        mark_lvgl_build_failed("nav mark");
        return btn;
    }
    lv_obj_remove_style_all(mark);
    lv_obj_set_pos(mark, 8, 8);
    lv_obj_set_size(mark, 22, 22);
    lv_obj_set_style_radius(mark, 6, LV_PART_MAIN);
    lv_obj_set_style_bg_color(mark, lv_color_hex(0x145883), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(mark, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_t *mark_label = make_text(mark, item.icon, 0, 3, 22, font_14(), 0xffffff, LV_LABEL_LONG_CLIP);
    if (mark_label) {
        lv_obj_set_style_text_align(mark_label, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
    }

    make_text(btn, item.text, 40, 9, 132, font_14(), active ? 0x0b344f : 0x36505f, LV_LABEL_LONG_DOT);
    return btn;
}

static lv_obj_t *make_product_shell(lv_obj_t *scr,
                                    const char *active_id,
                                    const char *title,
                                    const char *subtitle)
{
    if (!scr) {
        mark_lvgl_build_failed("shell screen");
        return nullptr;
    }
    style_product_screen(scr);

    lv_obj_t *nav = lv_obj_create(scr);
    if (!nav) {
        mark_lvgl_build_failed("nav shell");
        return nullptr;
    }
    lv_obj_set_pos(nav, 0, 0);
    lv_obj_set_size(nav, 216, SCREEN_H);
    lv_obj_clear_flag(nav, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_color(nav, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_border_color(nav, lv_color_hex(0xd9e2e8), LV_PART_MAIN);
    lv_obj_set_style_border_width(nav, 0, LV_PART_MAIN);
    lv_obj_set_style_border_side(nav, LV_BORDER_SIDE_RIGHT, LV_PART_MAIN);
    lv_obj_set_style_radius(nav, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_all(nav, 0, LV_PART_MAIN);

    lv_obj_t *logo = lv_image_create(nav);
    if (logo) {
        lv_image_set_src(logo, &img_competition_logo);
        lv_obj_set_pos(logo, 18, 16);
    } else {
        mark_lvgl_build_failed("logo");
    }

    make_text(nav, "智能健身镜", 18, 66, 174, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(nav, "姿态纠正训练系统", 18, 92, 174, font_14(), 0x687783, LV_LABEL_LONG_DOT);

    int32_t y = 120;
    for (size_t i = 0; i < sizeof(PRODUCT_NAV) / sizeof(PRODUCT_NAV[0]); i++) {
        make_nav_button(nav, y, PRODUCT_NAV[i], active_id);
        y += 42;
    }

    lv_obj_t *foot = make_card(nav, 14, SCREEN_H - 96, 188, 78);
    if (foot) {
        lv_obj_set_style_shadow_width(foot, 0, LV_PART_MAIN);
    }
    make_text(foot, "ESP32-P4 + PC姿态", 0, 0, 160, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    lbl_menu_music = make_text(foot, "音乐: --", 0, 28, 160, font_14(), 0x314551, LV_LABEL_LONG_DOT);

    lv_obj_t *main = lv_obj_create(scr);
    if (!main) {
        mark_lvgl_build_failed("main shell");
        return nullptr;
    }
    lv_obj_set_pos(main, 216, 0);
    lv_obj_set_size(main, SCREEN_W - 216, SCREEN_H);
    lv_obj_clear_flag(main, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_style_bg_color(main, lv_color_hex(0xf4f7f8), LV_PART_MAIN);
    lv_obj_set_style_border_width(main, 0, LV_PART_MAIN);
    lv_obj_set_style_radius(main, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_all(main, 0, LV_PART_MAIN);

    make_text(main, title, 24, 22, 390, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(main, subtitle, 24, 50, 500, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    return main;
}

static void add_top_status(lv_obj_t *main, const char *a, const char *b, const char *c)
{
    make_pill(main, 430, 22, 112, a, 0x22a06b);
    make_pill(main, 552, 22, 112, b, 0x22a06b);
    make_pill(main, 674, 22, 110, c, 0xd9822b);
}

static uint32_t get_frame_stride_pixels(uint32_t width, uint32_t height, size_t len)
{
    if (height == 0) {
        return width;
    }

    const uint32_t stride = (uint32_t)(len / height / 2);
    return stride >= width ? stride : width;
}

static void copy_rgb565_compact(uint8_t *dst_buf, const uint8_t *src_buf, uint32_t src_w, uint32_t src_h, size_t src_len)
{
    if (!dst_buf || !src_buf || src_w == 0 || src_h == 0) {
        return;
    }

    const uint32_t src_stride = get_frame_stride_pixels(src_w, src_h, src_len);
    const uint16_t *src = reinterpret_cast<const uint16_t *>(src_buf);
    uint16_t *dst = reinterpret_cast<uint16_t *>(dst_buf);

    for (uint32_t y = 0; y < src_h; y++) {
        memcpy(dst + y * src_w, src + y * src_stride, src_w * 2);
    }
}

static void compose_preview_rgb565(uint8_t *dst_buf,
                                   uint32_t dst_w,
                                   uint32_t dst_h,
                                   const uint8_t *src_buf,
                                   uint32_t src_w,
                                   uint32_t src_h,
                                   size_t src_len)
{
    if (!dst_buf || !src_buf || dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0) {
        return;
    }

    const uint32_t src_stride = get_frame_stride_pixels(src_w, src_h, src_len);
    const uint16_t *src = reinterpret_cast<const uint16_t *>(src_buf);
    uint16_t *dst = reinterpret_cast<uint16_t *>(dst_buf);

    for (uint32_t y = 0; y < dst_h; y++) {
        const uint32_t sy = (uint32_t)(((uint64_t)y * src_h) / dst_h);
        const uint16_t *src_row = src + sy * src_stride;
        uint16_t *dst_row = dst + y * dst_w;
        for (uint32_t x = 0; x < dst_w; x++) {
            const uint32_t sx = (uint32_t)(((uint64_t)x * src_w) / dst_w);
            dst_row[x] = src_row[sx];
        }
    }
}

static esp_err_t save_face_name(uint16_t id, const char *name)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("face_nm", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }

    char key[12];
    snprintf(key, sizeof(key), "u%u", (unsigned)id);
    err = nvs_set_str(h, key, name);
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    return err;
}

static bool load_face_name(uint16_t id, char *out, size_t outlen)
{
    nvs_handle_t h;
    if (nvs_open("face_nm", NVS_READONLY, &h) != ESP_OK) {
        return false;
    }

    char key[12];
    snprintf(key, sizeof(key), "u%u", (unsigned)id);
    size_t req = outlen;
    esp_err_t err = nvs_get_str(h, key, out, &req);
    nvs_close(h);
    return err == ESP_OK;
}

static esp_err_t erase_face_name(uint16_t id)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("face_nm", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }

    char key[12];
    snprintf(key, sizeof(key), "u%u", (unsigned)id);
    err = nvs_erase_key(h, key);
    if (err == ESP_ERR_NVS_NOT_FOUND) {
        err = ESP_OK;
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    return err;
}

static esp_err_t erase_all_face_names(void)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("face_nm", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }

    err = nvs_erase_all(h);
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    return err;
}

static void load_wifi_credentials(void)
{
    strlcpy(s_wifi_ssid, CONFIG_PC_POSE_WIFI_SSID, sizeof(s_wifi_ssid));
    strlcpy(s_wifi_password, CONFIG_PC_POSE_WIFI_PASSWORD, sizeof(s_wifi_password));

    nvs_handle_t h;
    if (nvs_open("wifi_cfg", NVS_READONLY, &h) != ESP_OK) {
        return;
    }

    size_t len = sizeof(s_wifi_ssid);
    if (nvs_get_str(h, "ssid", s_wifi_ssid, &len) != ESP_OK || s_wifi_ssid[0] == '\0') {
        strlcpy(s_wifi_ssid, CONFIG_PC_POSE_WIFI_SSID, sizeof(s_wifi_ssid));
    }
    len = sizeof(s_wifi_password);
    if (nvs_get_str(h, "pass", s_wifi_password, &len) != ESP_OK) {
        strlcpy(s_wifi_password, CONFIG_PC_POSE_WIFI_PASSWORD, sizeof(s_wifi_password));
    }
    nvs_close(h);
}

static esp_err_t save_wifi_credentials(const char *ssid, const char *password)
{
    if (!ssid || ssid[0] == '\0') {
        return ESP_ERR_INVALID_ARG;
    }

    nvs_handle_t h;
    esp_err_t err = nvs_open("wifi_cfg", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }
    err = nvs_set_str(h, "ssid", ssid);
    if (err == ESP_OK) {
        err = nvs_set_str(h, "pass", password ? password : "");
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    return err;
}

static esp_err_t erase_wifi_credentials(void)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("wifi_cfg", NVS_READWRITE, &h);
    if (err == ESP_ERR_NVS_NOT_FOUND) {
        return ESP_OK;
    }
    if (err != ESP_OK) {
        return err;
    }

    err = nvs_erase_all(h);
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    return err;
}

static void load_auth_account(void)
{
    s_auth_saved_user[0] = '\0';
    s_auth_saved_pass[0] = '\0';
    s_auth_has_password.store(false);

    nvs_handle_t h;
    if (nvs_open("auth_cfg", NVS_READONLY, &h) != ESP_OK) {
        return;
    }

    size_t len = sizeof(s_auth_saved_user);
    const esp_err_t user_err = nvs_get_str(h, "user", s_auth_saved_user, &len);
    len = sizeof(s_auth_saved_pass);
    const esp_err_t pass_err = nvs_get_str(h, "pass", s_auth_saved_pass, &len);
    nvs_close(h);

    s_auth_has_password.store(user_err == ESP_OK && pass_err == ESP_OK &&
                              s_auth_saved_user[0] != '\0' && s_auth_saved_pass[0] != '\0');
}

static esp_err_t save_auth_account(const char *user, const char *password)
{
    if (!user || user[0] == '\0' || !password || password[0] == '\0') {
        return ESP_ERR_INVALID_ARG;
    }

    nvs_handle_t h;
    esp_err_t err = nvs_open("auth_cfg", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }
    err = nvs_set_str(h, "user", user);
    if (err == ESP_OK) {
        err = nvs_set_str(h, "pass", password);
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);

    if (err == ESP_OK) {
        strlcpy(s_auth_saved_user, user, sizeof(s_auth_saved_user));
        strlcpy(s_auth_saved_pass, password, sizeof(s_auth_saved_pass));
        s_auth_has_password.store(true);
    }
    return err;
}

static esp_err_t erase_auth_account(void)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("auth_cfg", NVS_READWRITE, &h);
    if (err == ESP_ERR_NVS_NOT_FOUND) {
        err = ESP_OK;
    } else if (err == ESP_OK) {
        err = nvs_erase_all(h);
        if (err == ESP_OK) {
            err = nvs_commit(h);
        }
        nvs_close(h);
    }

    if (err == ESP_OK) {
        s_auth_saved_user[0] = '\0';
        s_auth_saved_pass[0] = '\0';
        s_auth_user[0] = '\0';
        s_auth_has_password.store(false);
        s_auth_face_login_active.store(false);
        s_auth_unlock_pending.store(false);
    }
    return err;
}

static void user_profile_set_defaults(user_profile_t *profile)
{
    if (!profile) {
        return;
    }
    memset(profile, 0, sizeof(*profile));
    strlcpy(profile->name, "用户", sizeof(profile->name));
    strlcpy(profile->gender, "男", sizeof(profile->gender));
    strlcpy(profile->goal, "塑形", sizeof(profile->goal));
    strlcpy(profile->focus, "全身", sizeof(profile->focus));
    profile->height_cm = 170;
    profile->weight_kg = 65;
    profile->minutes_per_day = 20;
    profile->complete = false;
}

static bool text_contains(const char *value, const char *needle)
{
    return value && needle && strstr(value, needle) != nullptr;
}

static int user_recommended_training_index(void)
{
    if (text_contains(s_user_profile.focus, "核心")) {
        return 5;
    }
    if (text_contains(s_user_profile.focus, "上肢")) {
        return 1;
    }
    if (text_contains(s_user_profile.focus, "下肢") || text_contains(s_user_profile.goal, "力量") ||
        text_contains(s_user_profile.goal, "塑形")) {
        return 0;
    }
    if (text_contains(s_user_profile.goal, "减脂")) {
        return 1;
    }
    if (text_contains(s_user_profile.goal, "康复")) {
        return 5;
    }
    return 0;
}

static void user_profile_summary(char *out, size_t out_len)
{
    if (!out || out_len == 0) {
        return;
    }
    if (!s_user_profile.complete) {
        snprintf(out, out_len, "资料: 未完善");
        return;
    }
    snprintf(out,
             out_len,
             "%s · %s · %s · %umin/天",
             s_user_profile.name,
             s_user_profile.goal,
             s_user_profile.focus,
             (unsigned)s_user_profile.minutes_per_day);
}

static void load_user_profile(void)
{
    user_profile_set_defaults(&s_user_profile);

    nvs_handle_t h;
    if (nvs_open("user_prof", NVS_READONLY, &h) != ESP_OK) {
        return;
    }

    uint8_t complete = 0;
    (void)nvs_get_u8(h, "complete", &complete);
    size_t len = sizeof(s_user_profile.name);
    (void)nvs_get_str(h, "name", s_user_profile.name, &len);
    len = sizeof(s_user_profile.gender);
    (void)nvs_get_str(h, "gender", s_user_profile.gender, &len);
    len = sizeof(s_user_profile.goal);
    (void)nvs_get_str(h, "goal", s_user_profile.goal, &len);
    len = sizeof(s_user_profile.focus);
    (void)nvs_get_str(h, "focus", s_user_profile.focus, &len);
    int32_t value = 0;
    if (nvs_get_i32(h, "height", &value) == ESP_OK && value > 0) {
        s_user_profile.height_cm = (uint16_t)value;
    }
    if (nvs_get_i32(h, "weight", &value) == ESP_OK && value > 0) {
        s_user_profile.weight_kg = (uint16_t)value;
    }
    if (nvs_get_i32(h, "minutes", &value) == ESP_OK && value > 0) {
        s_user_profile.minutes_per_day = (uint16_t)value;
    }
    nvs_close(h);

    s_user_profile.complete = complete != 0 && s_user_profile.name[0] != '\0';
    if (s_user_profile.gender[0] == '\0') {
        strlcpy(s_user_profile.gender, "男", sizeof(s_user_profile.gender));
    }
    if (s_user_profile.goal[0] == '\0') {
        strlcpy(s_user_profile.goal, "塑形", sizeof(s_user_profile.goal));
    }
    if (s_user_profile.focus[0] == '\0') {
        strlcpy(s_user_profile.focus, "全身", sizeof(s_user_profile.focus));
    }
}

static esp_err_t save_user_profile(const user_profile_t *profile)
{
    if (!profile || profile->name[0] == '\0') {
        return ESP_ERR_INVALID_ARG;
    }

    nvs_handle_t h;
    esp_err_t err = nvs_open("user_prof", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }
    err = nvs_set_u8(h, "complete", profile->complete ? 1 : 0);
    if (err == ESP_OK) {
        err = nvs_set_str(h, "name", profile->name);
    }
    if (err == ESP_OK) {
        err = nvs_set_str(h, "gender", profile->gender);
    }
    if (err == ESP_OK) {
        err = nvs_set_str(h, "goal", profile->goal);
    }
    if (err == ESP_OK) {
        err = nvs_set_str(h, "focus", profile->focus);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "height", profile->height_cm);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "weight", profile->weight_kg);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "minutes", profile->minutes_per_day);
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);

    if (err == ESP_OK) {
        s_user_profile = *profile;
    }
    return err;
}

static void load_training_record(void)
{
    memset(&s_training_record, 0, sizeof(s_training_record));
    strlcpy(s_training_record.last_training, "--", sizeof(s_training_record.last_training));
    strlcpy(s_training_record.last_time, "--", sizeof(s_training_record.last_time));
    strlcpy(s_training_record.last_tip, "完成训练后生成记录。", sizeof(s_training_record.last_tip));

    nvs_handle_t h;
    if (nvs_open("train_rec", NVS_READONLY, &h) != ESP_OK) {
        return;
    }

    (void)nvs_get_u32(h, "sessions", &s_training_record.sessions);
    int32_t value = 0;
    if (nvs_get_i32(h, "score", &value) == ESP_OK) {
        s_training_record.last_score = value;
    }
    if (nvs_get_i32(h, "depth", &value) == ESP_OK) {
        s_training_record.last_score_depth = value;
    }
    if (nvs_get_i32(h, "knee", &value) == ESP_OK) {
        s_training_record.last_score_knee = value;
    }
    if (nvs_get_i32(h, "hip", &value) == ESP_OK) {
        s_training_record.last_score_hip = value;
    }
    if (nvs_get_i32(h, "torso", &value) == ESP_OK) {
        s_training_record.last_score_torso = value;
    }
    if (nvs_get_i32(h, "balance", &value) == ESP_OK) {
        s_training_record.last_score_balance = value;
    }
    if (nvs_get_i32(h, "track", &value) == ESP_OK) {
        s_training_record.last_score_track = value;
    }
    if (nvs_get_i32(h, "count", &value) == ESP_OK) {
        s_training_record.last_count = value;
    }
    if (nvs_get_i32(h, "target", &value) == ESP_OK) {
        s_training_record.last_target = value;
    }
    if (nvs_get_i32(h, "elapsed", &value) == ESP_OK && value >= 0) {
        s_training_record.last_elapsed_ms = (uint32_t)value;
    }
    if (nvs_get_i32(h, "total_cnt", &value) == ESP_OK && value >= 0) {
        s_training_record.total_count = (uint32_t)value;
    }
    if (nvs_get_i32(h, "total_ms", &value) == ESP_OK && value >= 0) {
        s_training_record.total_elapsed_ms = (uint32_t)value;
    }
    size_t len = sizeof(s_training_record.last_training);
    (void)nvs_get_str(h, "training", s_training_record.last_training, &len);
    len = sizeof(s_training_record.last_time);
    (void)nvs_get_str(h, "time", s_training_record.last_time, &len);
    len = sizeof(s_training_record.last_tip);
    (void)nvs_get_str(h, "tip", s_training_record.last_tip, &len);
    nvs_close(h);
}

static esp_err_t save_training_record_to_nvs(void)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("train_rec", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }
    err = nvs_set_u32(h, "sessions", s_training_record.sessions);
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "score", s_training_record.last_score);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "depth", s_training_record.last_score_depth);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "knee", s_training_record.last_score_knee);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "hip", s_training_record.last_score_hip);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "torso", s_training_record.last_score_torso);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "balance", s_training_record.last_score_balance);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "track", s_training_record.last_score_track);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "count", s_training_record.last_count);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "target", s_training_record.last_target);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "elapsed", (int32_t)s_training_record.last_elapsed_ms);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "total_cnt", (int32_t)s_training_record.total_count);
    }
    if (err == ESP_OK) {
        err = nvs_set_i32(h, "total_ms", (int32_t)s_training_record.total_elapsed_ms);
    }
    if (err == ESP_OK) {
        err = nvs_set_str(h, "training", s_training_record.last_training);
    }
    if (err == ESP_OK) {
        err = nvs_set_str(h, "time", s_training_record.last_time);
    }
    if (err == ESP_OK) {
        err = nvs_set_str(h, "tip", s_training_record.last_tip);
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    return err;
}

static void training_history_sanitize_copy(char *dst, size_t dst_len, const char *src)
{
    if (!dst || dst_len == 0) {
        return;
    }
    size_t out = 0;
    if (src) {
        for (size_t i = 0; src[i] != '\0' && out + 1 < dst_len; ++i) {
            char ch = src[i];
            if (ch == '|' || ch == '\r' || ch == '\n' || ch == '\t') {
                ch = ' ';
            }
            dst[out++] = ch;
        }
    }
    dst[out] = '\0';
}

static bool training_history_serialize(const training_history_item_t &item, char *out, size_t out_len)
{
    if (!out || out_len == 0 || !item.valid) {
        return false;
    }
    char tip[sizeof(item.tip)];
    char reps[sizeof(item.reps)];
    training_history_sanitize_copy(tip, sizeof(tip), item.tip);
    training_history_sanitize_copy(reps, sizeof(reps), item.reps);
    const int written = snprintf(out,
                                 out_len,
                                 "%s|%s|%s|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%u|%s|%s",
                                 item.source,
                                 item.time,
                                 item.action,
                                 item.count,
                                 item.target,
                                 item.score,
                                 item.rep_score,
                                 item.score_depth,
                                 item.score_knee,
                                 item.score_hip,
                                 item.score_torso,
                                 item.score_balance,
                                 item.score_track,
                                 (unsigned)item.elapsed_ms,
                                 tip,
                                 reps);
    return written > 0 && (size_t)written < out_len;
}

static bool training_history_deserialize(const char *line, training_history_item_t *out)
{
    if (!line || !out) {
        return false;
    }
    char buf[560];
    strlcpy(buf, line, sizeof(buf));
    char *fields[16] = {};
    int n = 0;
    char *cursor = buf;
    while (cursor && n < 16) {
        fields[n++] = cursor;
        char *sep = strchr(cursor, '|');
        if (!sep) {
            break;
        }
        *sep = '\0';
        cursor = sep + 1;
    }
    if (n < 15) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    out->valid = true;
    strlcpy(out->source, fields[0], sizeof(out->source));
    strlcpy(out->time, fields[1], sizeof(out->time));
    strlcpy(out->action, fields[2], sizeof(out->action));
    out->count = atoi(fields[3]);
    out->target = atoi(fields[4]);
    out->score = atoi(fields[5]);
    out->rep_score = atoi(fields[6]);
    out->score_depth = atoi(fields[7]);
    out->score_knee = atoi(fields[8]);
    out->score_hip = atoi(fields[9]);
    out->score_torso = atoi(fields[10]);
    out->score_balance = atoi(fields[11]);
    out->score_track = atoi(fields[12]);
    out->elapsed_ms = (uint32_t)strtoul(fields[13], nullptr, 10);
    strlcpy(out->tip, fields[14], sizeof(out->tip));
    if (n >= 16) {
        strlcpy(out->reps, fields[15], sizeof(out->reps));
    }
    return out->source[0] != '\0' && out->action[0] != '\0';
}

static void load_training_history(void)
{
    memset(s_training_history, 0, sizeof(s_training_history));
    s_training_history_count = 0;
    nvs_handle_t h;
    if (nvs_open("train_hist", NVS_READONLY, &h) != ESP_OK) {
        return;
    }
    int32_t count = 0;
    (void)nvs_get_i32(h, "count", &count);
    count = std::max<int32_t>(0, std::min<int32_t>(TRAINING_HISTORY_CAPACITY, count));
    for (int i = 0; i < count; ++i) {
        char key[8];
        snprintf(key, sizeof(key), "h%02d", i);
        char line[560];
        size_t len = sizeof(line);
        if (nvs_get_str(h, key, line, &len) == ESP_OK &&
            training_history_deserialize(line, &s_training_history[s_training_history_count])) {
            s_training_history_count++;
        }
    }
    nvs_close(h);
}

static esp_err_t save_training_history_to_nvs(void)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("train_hist", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }
    err = nvs_set_i32(h, "count", s_training_history_count);
    for (int i = 0; err == ESP_OK && i < TRAINING_HISTORY_CAPACITY; ++i) {
        char key[8];
        snprintf(key, sizeof(key), "h%02d", i);
        if (i < s_training_history_count && s_training_history[i].valid) {
            char line[560];
            if (!training_history_serialize(s_training_history[i], line, sizeof(line))) {
                err = ESP_ERR_INVALID_ARG;
                break;
            }
            err = nvs_set_str(h, key, line);
        } else {
            (void)nvs_erase_key(h, key);
        }
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    return err;
}

static void training_history_append(const training_history_item_t &item)
{
    if (!item.valid || item.action[0] == '\0') {
        return;
    }
    const int next_count = std::min(TRAINING_HISTORY_CAPACITY, s_training_history_count + 1);
    for (int i = next_count - 1; i > 0; --i) {
        s_training_history[i] = s_training_history[i - 1];
    }
    s_training_history[0] = item;
    s_training_history[0].valid = true;
    s_training_history_count = next_count;
    const esp_err_t err = save_training_history_to_nvs();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "save training history failed: %s", esp_err_to_name(err));
    }
}

static const char *training_history_source_text(const char *source)
{
    return (source && strcmp(source, "offline") == 0) ? "离线" : "实时";
}

static bool training_history_date_prefix(char *out, size_t out_len)
{
    if (!out || out_len == 0) {
        return false;
    }
    time_t now = time(nullptr);
    struct tm tm_now = {};
    localtime_r(&now, &tm_now);
    if (tm_now.tm_year < 120) {
        out[0] = '\0';
        return false;
    }
    snprintf(out,
             out_len,
             "%04d-%02d-%02d",
             tm_now.tm_year + 1900,
             tm_now.tm_mon + 1,
             tm_now.tm_mday);
    return true;
}

static bool lock_models(TickType_t ticks)
{
    return s_model_mutex && xSemaphoreTake(s_model_mutex, ticks) == pdTRUE;
}

static void unlock_models(void)
{
    if (s_model_mutex) {
        xSemaphoreGive(s_model_mutex);
    }
}

static void log_pose_results(const char *prefix, const std::vector<pose_result_t> &results)
{
    ESP_LOGI(TAG, "%s results=%d", prefix, (int)results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        const pose_result_t &res = results[i];
        ESP_LOGI(TAG,
                 "%s person[%u] score=%.3f box=(%d,%d,%d,%d)",
                 prefix,
                 (unsigned)i,
                 res.score,
                 res.x1,
                 res.y1,
                 res.x2,
                 res.y2);
        for (size_t k = 0; k < res.keypoints.size() && k < POSE_KPTS; ++k) {
            const pose_keypoint_t &kp = res.keypoints[k];
            ESP_LOGI(TAG,
                     "%s person[%u] %s=(%d,%d) score=%.3f",
                     prefix,
                     (unsigned)i,
                     POSE_KEYPOINT_NAMES[k],
                     kp.x,
                     kp.y,
                     kp.score);
        }
    }
}

static void run_static_pose_self_test(void)
{
    const size_t jpg_len = (size_t)(test_person_jpg_end - test_person_jpg_start);
    if (jpg_len == 0) {
        ESP_LOGW(TAG, "Static pose test image is empty");
        return;
    }

    ESP_LOGI(TAG, "Static pose self-test start: jpeg_bytes=%u", (unsigned)jpg_len);
    const int64_t decode_start = esp_timer_get_time();
    dl::image::jpeg_img_t jpeg_img = {
        .data = (void *)test_person_jpg_start,
        .data_len = jpg_len,
    };
    dl::image::img_t img = dl::image::sw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB888);
    const int64_t decode_end = esp_timer_get_time();
    if (!img.data || img.width == 0 || img.height == 0) {
        ESP_LOGE(TAG, "Static pose self-test JPEG decode failed");
        return;
    }

    ESP_LOGI(TAG,
             "Static pose self-test image decoded: %ux%u in %.2f ms",
             (unsigned)img.width,
             (unsigned)img.height,
             (decode_end - decode_start) / 1000.0);

    HumanPoseDetect pose_det;
    if (!pose_det.ready()) {
        ESP_LOGE(TAG, "Static pose self-test model init failed: %s", pose_det.error());
        heap_caps_free(img.data);
        return;
    }

    const int64_t infer_start = esp_timer_get_time();
    const std::vector<pose_result_t> &results = pose_det.run(img);
    const int64_t infer_end = esp_timer_get_time();

    ESP_LOGI(TAG,
             "Static pose self-test infer total: %.2f ms quality=%.2f valid=%d avg=%.3f",
             (infer_end - infer_start) / 1000.0,
             (double)pose_det.last_quality(),
             pose_det.last_valid_count(),
             (double)pose_det.last_avg_score());
    log_pose_results("static_pose", results);

    heap_caps_free(img.data);
}

static void reset_pc_pose_smooth(void);
static void clear_pose_tracking_roi(void);

static void release_pose_model(void)
{
    if (s_pose_det) {
        delete s_pose_det;
        s_pose_det = nullptr;
    }
    if (s_body_det) {
        delete s_body_det;
        s_body_det = nullptr;
    }
    s_pose_input_variant = -1;
    s_body_roi_reacquire_requested.store(false);
    s_last_pc_pose_success_us.store(0);
    reset_pc_pose_smooth();
    clear_pose_tracking_roi();
}

static void release_face_models(void)
{
    if (s_rec) {
        delete s_rec;
        s_rec = nullptr;
    }
    if (s_det) {
        delete s_det;
        s_det = nullptr;
    }
}

static bool ensure_face_models(void)
{
    if (s_det && s_rec) {
        return true;
    }

    char db_path[64];
    snprintf(db_path, sizeof(db_path), "%s/face.db", BSP_SPIFFS_MOUNT_POINT);
    push_ui_msg("正在加载人脸模型...");
    if (!s_det) {
        s_det = new HumanFaceDetect();
    }
    if (!s_rec) {
        s_rec = new HumanFaceRecognizer(db_path);
    }

    if (!s_det || !s_rec) {
        push_ui_msg("人脸模型分配失败");
        return false;
    }
    s_det->set_score_thr(0.35f, 0);
    s_det->set_score_thr(0.35f, 1);
    s_db_count.store(s_rec->get_num_feats());
    push_ui_msg("人脸模型就绪");
    return true;
}

static void release_models_if_idle(bool pose, bool face)
{
    if (!lock_models(0)) {
        ESP_LOGI(TAG, "Model release deferred: worker is busy");
        return;
    }

    if (pose) {
        release_pose_model();
    }
    if (face) {
        release_face_models();
    }
    unlock_models();
}

static void resize_rgb565_nearest(uint8_t *dst,
                                  uint32_t dst_w,
                                  uint32_t dst_h,
                                  const uint8_t *src,
                                  uint32_t src_w,
                                  uint32_t src_h,
                                  bool rotate_180 = false)
{
    if (!dst || !src || dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0) {
        return;
    }

    const uint16_t *src_px = reinterpret_cast<const uint16_t *>(src);
    uint16_t *dst_px = reinterpret_cast<uint16_t *>(dst);
    for (uint32_t y = 0; y < dst_h; y++) {
        const uint32_t sy = (uint32_t)(((uint64_t)y * src_h) / dst_h);
        const uint16_t *src_row = src_px + sy * src_w;
        uint16_t *dst_row = dst_px + y * dst_w;
        for (uint32_t x = 0; x < dst_w; x++) {
            const uint32_t sx = (uint32_t)(((uint64_t)x * src_w) / dst_w);
            if (rotate_180) {
                dst_row[x] = src_px[(src_h - 1 - sy) * src_w + (src_w - 1 - sx)];
            } else {
                dst_row[x] = src_row[sx];
            }
        }
    }
}

static void resize_rgb565_crop_nearest_strided(uint8_t *dst,
                                               uint32_t dst_w,
                                               uint32_t dst_h,
                                               const uint8_t *src,
                                               uint32_t src_w,
                                               uint32_t src_h,
                                               size_t src_len,
                                               uint32_t crop_x,
                                               uint32_t crop_y,
                                               uint32_t crop_w,
                                               uint32_t crop_h,
                                               bool rotate_180 = false)
{
    if (!dst || !src || dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 || crop_w == 0 || crop_h == 0) {
        return;
    }

    crop_x = std::min(crop_x, src_w - 1);
    crop_y = std::min(crop_y, src_h - 1);
    crop_w = std::min(crop_w, src_w - crop_x);
    crop_h = std::min(crop_h, src_h - crop_y);

    const uint32_t src_stride = get_frame_stride_pixels(src_w, src_h, src_len);
    const uint16_t *src_px = reinterpret_cast<const uint16_t *>(src);
    uint16_t *dst_px = reinterpret_cast<uint16_t *>(dst);
    for (uint32_t y = 0; y < dst_h; y++) {
        const uint32_t rel_sy = (uint32_t)(((uint64_t)y * crop_h) / dst_h);
        const uint32_t sy = rotate_180 ? (crop_y + crop_h - 1 - rel_sy) : (crop_y + rel_sy);
        uint16_t *dst_row = dst_px + y * dst_w;
        for (uint32_t x = 0; x < dst_w; x++) {
            const uint32_t rel_sx = (uint32_t)(((uint64_t)x * crop_w) / dst_w);
            const uint32_t sx = rotate_180 ? (crop_x + crop_w - 1 - rel_sx) : (crop_x + rel_sx);
            dst_row[x] = src_px[sy * src_stride + sx];
        }
    }
}

static uint32_t map_resized_pixel_center(uint32_t dst_i, uint32_t dst_n, uint32_t src_n)
{
    if (dst_n == 0 || src_n == 0) {
        return 0;
    }
    const float src_f = (((float)dst_i + 0.5f) * (float)src_n / (float)dst_n) - 0.5f;
    const int src_i = (int)lroundf(src_f);
    return (uint32_t)std::max(0, std::min((int)src_n - 1, src_i));
}

static void make_pose_letterbox_region(uint32_t src_w,
                                       uint32_t src_h,
                                       uint32_t dst_w,
                                       uint32_t dst_h,
                                       uint32_t *box_x,
                                       uint32_t *box_y,
                                       uint32_t *box_w,
                                       uint32_t *box_h)
{
    if (!box_x || !box_y || !box_w || !box_h || src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0) {
        return;
    }

    uint32_t fit_w = dst_w;
    uint32_t fit_h = (uint32_t)(((uint64_t)src_h * dst_w + src_w / 2) / src_w);
    if (fit_h == 0) {
        fit_h = 1;
    }
    if (fit_h > dst_h) {
        fit_h = dst_h;
        fit_w = (uint32_t)(((uint64_t)src_w * dst_h + src_h / 2) / src_h);
        if (fit_w == 0) {
            fit_w = 1;
        }
        fit_w = std::min(fit_w, dst_w);
    }

    *box_x = (dst_w - fit_w) / 2;
    *box_y = (dst_h - fit_h) / 2;
    *box_w = fit_w;
    *box_h = fit_h;
}

static void resize_rgb565_letterbox_nearest_strided(uint8_t *dst,
                                                    uint32_t dst_w,
                                                    uint32_t dst_h,
                                                    const uint8_t *src,
                                                    uint32_t src_w,
                                                    uint32_t src_h,
                                                    size_t src_len,
                                                    uint32_t box_x,
                                                    uint32_t box_y,
                                                    uint32_t box_w,
                                                    uint32_t box_h,
                                                    bool rotate_180 = false)
{
    if (!dst || !src || dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 || box_w == 0 || box_h == 0) {
        return;
    }

    memset(dst, 0, (size_t)dst_w * (size_t)dst_h * 2);
    box_x = std::min(box_x, dst_w - 1);
    box_y = std::min(box_y, dst_h - 1);
    box_w = std::min(box_w, dst_w - box_x);
    box_h = std::min(box_h, dst_h - box_y);

    const uint32_t src_stride = get_frame_stride_pixels(src_w, src_h, src_len);
    const uint16_t *src_px = reinterpret_cast<const uint16_t *>(src);
    uint16_t *dst_px = reinterpret_cast<uint16_t *>(dst);
    for (uint32_t y = 0; y < box_h; y++) {
        const uint32_t sy0 = map_resized_pixel_center(y, box_h, src_h);
        const uint32_t sy = rotate_180 ? (src_h - 1 - sy0) : sy0;
        uint16_t *dst_row = dst_px + (box_y + y) * dst_w + box_x;
        for (uint32_t x = 0; x < box_w; x++) {
            const uint32_t sx0 = map_resized_pixel_center(x, box_w, src_w);
            const uint32_t sx = rotate_180 ? (src_w - 1 - sx0) : sx0;
            dst_row[x] = src_px[sy * src_stride + sx];
        }
    }
}

static bool ensure_pc_pose_ppa_srm(void)
{
    if (s_pc_pose_ppa_srm) {
        return true;
    }
    if (s_pc_pose_ppa_disabled) {
        return false;
    }

    ppa_client_config_t cfg = {};
    cfg.oper_type = PPA_OPERATION_SRM;
    cfg.max_pending_trans_num = 1;
    cfg.data_burst_length = PPA_DATA_BURST_LENGTH_128;

    const esp_err_t err = ppa_register_client(&cfg, &s_pc_pose_ppa_srm);
    if (err != ESP_OK) {
        s_pc_pose_ppa_disabled = true;
        ESP_LOGW(TAG, "PC pose PPA SRM unavailable: %s", esp_err_to_name(err));
        return false;
    }
    ESP_LOGI(TAG, "PPA SRM resize active");
    return true;
}

static bool resize_rgb565_crop_ppa(uint8_t *dst,
                                   uint32_t dst_w,
                                   uint32_t dst_h,
                                   const uint8_t *src,
                                   uint32_t src_w,
                                   uint32_t src_h,
                                   size_t src_len,
                                   uint32_t crop_x,
                                   uint32_t crop_y,
                                   uint32_t crop_w,
                                   uint32_t crop_h,
                                   bool rotate_180 = false)
{
    if (!dst || !src || dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 || crop_w == 0 || crop_h == 0) {
        return false;
    }
    if (!ensure_pc_pose_ppa_srm()) {
        return false;
    }

    crop_x = std::min(crop_x, src_w - 1);
    crop_y = std::min(crop_y, src_h - 1);
    crop_w = std::min(crop_w, src_w - crop_x);
    crop_h = std::min(crop_h, src_h - crop_y);

    const uint32_t src_stride = get_frame_stride_pixels(src_w, src_h, src_len);
    const size_t out_size = (size_t)dst_w * (size_t)dst_h * 2;
    ppa_srm_oper_config_t cfg = {};
    cfg.in.buffer = src;
    cfg.in.pic_w = src_stride;
    cfg.in.pic_h = src_h;
    cfg.in.block_w = crop_w;
    cfg.in.block_h = crop_h;
    cfg.in.block_offset_x = crop_x;
    cfg.in.block_offset_y = crop_y;
    cfg.in.srm_cm = PPA_SRM_COLOR_MODE_RGB565;
    cfg.out.buffer = dst;
    cfg.out.buffer_size = out_size;
    cfg.out.pic_w = dst_w;
    cfg.out.pic_h = dst_h;
    cfg.out.block_offset_x = 0;
    cfg.out.block_offset_y = 0;
    cfg.out.srm_cm = PPA_SRM_COLOR_MODE_RGB565;
    cfg.rotation_angle = rotate_180 ? PPA_SRM_ROTATION_ANGLE_180 : PPA_SRM_ROTATION_ANGLE_0;
    cfg.scale_x = (float)dst_w / (float)crop_w;
    cfg.scale_y = (float)dst_h / (float)crop_h;
    cfg.mode = PPA_TRANS_MODE_BLOCKING;

    const esp_err_t err = ppa_do_scale_rotate_mirror(s_pc_pose_ppa_srm, &cfg);
    if (err != ESP_OK) {
        static int64_t last_warn_us = 0;
        const int64_t now_us = esp_timer_get_time();
        if (now_us - last_warn_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
            last_warn_us = now_us;
            ESP_LOGW(TAG, "PC pose PPA resize failed: %s; fallback CPU", esp_err_to_name(err));
        }
        return false;
    }
    return true;
}

static bool resize_rgb565_letterbox_ppa(uint8_t *dst,
                                        uint32_t dst_w,
                                        uint32_t dst_h,
                                        const uint8_t *src,
                                        uint32_t src_w,
                                        uint32_t src_h,
                                        size_t src_len,
                                        uint32_t box_x,
                                        uint32_t box_y,
                                        uint32_t box_w,
                                        uint32_t box_h,
                                        bool rotate_180 = false)
{
    if (!dst || !src || dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 || box_w == 0 || box_h == 0) {
        return false;
    }
    if (!ensure_pc_pose_ppa_srm()) {
        return false;
    }

    box_x = std::min(box_x, dst_w - 1);
    box_y = std::min(box_y, dst_h - 1);
    box_w = std::min(box_w, dst_w - box_x);
    box_h = std::min(box_h, dst_h - box_y);

    memset(dst, 0, (size_t)dst_w * (size_t)dst_h * 2);

    const uint32_t src_stride = get_frame_stride_pixels(src_w, src_h, src_len);
    const size_t out_size = (size_t)dst_w * (size_t)dst_h * 2;
    ppa_srm_oper_config_t cfg = {};
    cfg.in.buffer = src;
    cfg.in.pic_w = src_stride;
    cfg.in.pic_h = src_h;
    cfg.in.block_w = src_w;
    cfg.in.block_h = src_h;
    cfg.in.block_offset_x = 0;
    cfg.in.block_offset_y = 0;
    cfg.in.srm_cm = PPA_SRM_COLOR_MODE_RGB565;
    cfg.out.buffer = dst;
    cfg.out.buffer_size = out_size;
    cfg.out.pic_w = dst_w;
    cfg.out.pic_h = dst_h;
    cfg.out.block_offset_x = box_x;
    cfg.out.block_offset_y = box_y;
    cfg.out.srm_cm = PPA_SRM_COLOR_MODE_RGB565;
    cfg.rotation_angle = rotate_180 ? PPA_SRM_ROTATION_ANGLE_180 : PPA_SRM_ROTATION_ANGLE_0;
    cfg.scale_x = (float)box_w / (float)src_w;
    cfg.scale_y = (float)box_h / (float)src_h;
    cfg.mode = PPA_TRANS_MODE_BLOCKING;

    const esp_err_t err = ppa_do_scale_rotate_mirror(s_pc_pose_ppa_srm, &cfg);
    if (err != ESP_OK) {
        static int64_t last_warn_us = 0;
        const int64_t now_us = esp_timer_get_time();
        if (now_us - last_warn_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
            last_warn_us = now_us;
            ESP_LOGW(TAG, "PC pose PPA letterbox failed: %s; fallback CPU", esp_err_to_name(err));
        }
        return false;
    }
    return true;
}

static bool detect_face_with_variants(const uint8_t *src,
                                      std::list<dl::detect::result_t> *dets,
                                      dl::image::img_t *img,
                                      const face_input_variant_t **variant_out)
{
    if (!src || !dets || !img || !s_ai_buf || !s_det || s_cam_w == 0 || s_cam_h == 0 || s_ai_w == 0 || s_ai_h == 0) {
        return false;
    }

    auto try_variant = [&](int idx) -> bool {
        const face_input_variant_t &variant = FACE_INPUT_VARIANTS[idx];
        resize_rgb565_nearest(s_ai_buf, s_ai_w, s_ai_h, src, s_cam_w, s_cam_h, variant.rotate_180);

        img->data = reinterpret_cast<uint16_t *>(s_ai_buf);
        img->width = (uint16_t)s_ai_w;
        img->height = (uint16_t)s_ai_h;
        img->pix_type = variant.pix_type;

        *dets = s_det->run(*img);
        if (!dets->empty()) {
            if (s_face_input_variant != idx) {
                s_face_input_variant = idx;
                push_ui_msg("Face input: %s", variant.name);
            }
            if (variant_out) {
                *variant_out = &variant;
            }
            return true;
        }
        return false;
    };

    if (s_face_input_variant >= 0 &&
        s_face_input_variant < (int)(sizeof(FACE_INPUT_VARIANTS) / sizeof(FACE_INPUT_VARIANTS[0])) &&
        try_variant(s_face_input_variant)) {
        return true;
    }

    for (int i = 0; i < (int)(sizeof(FACE_INPUT_VARIANTS) / sizeof(FACE_INPUT_VARIANTS[0])); i++) {
        if (i == s_face_input_variant) {
            continue;
        }
        if (try_variant(i)) {
            return true;
        }
    }

    if (variant_out) {
        *variant_out = nullptr;
    }
    return false;
}

static constexpr EventBits_t PC_POSE_WIFI_CONNECTED_BIT = BIT0;
static constexpr EventBits_t PC_POSE_WIFI_FAIL_BIT = BIT1;

static constexpr size_t PC_POSE_REQ_HEADER_BYTES = 16;
static constexpr size_t PC_POSE_RESP_HEADER_BYTES = 24;
static constexpr size_t PC_WEATHER_RESP_HEADER_BYTES = 12;
static constexpr size_t PC_VOICE_RESP_HEADER_BYTES = 20;
static constexpr size_t PC_VOICE_MAX_PCM_BYTES = 256 * 1024;
static constexpr size_t PC_POSE_FLOATS_PER_KPT = 4;
static constexpr uint8_t PC_POSE_REQ_MAGIC_RGB565[4] = {'P', '4', 'P', '1'};
static constexpr uint8_t PC_POSE_REQ_MAGIC_RGB332[4] = {'P', '4', 'P', '2'};
static constexpr uint8_t PC_POSE_REQ_MAGIC_JPEG[4] = {'P', '4', 'J', '1'};
static constexpr uint8_t PC_WEATHER_REQ_MAGIC[4] = {'P', '4', 'W', '1'};
static constexpr uint8_t PC_VOICE_REQ_MAGIC[4] = {'P', '4', 'V', '1'};
static constexpr uint8_t PC_POSE_RESP_MAGIC[4] = {'P', '4', 'R', '1'};
static constexpr uint8_t PC_WEATHER_RESP_MAGIC[4] = {'P', '4', 'W', '2'};
static constexpr uint8_t PC_VOICE_RESP_MAGIC[4] = {'P', '4', 'V', '2'};

static void set_weather_text(const char *text);
static void get_weather_text_copy(char *out, size_t out_len);

static void pc_pose_close_socket(void)
{
    if (s_pc_pose_sock >= 0) {
        shutdown(s_pc_pose_sock, 0);
        close(s_pc_pose_sock);
        s_pc_pose_sock = -1;
    }
    if (s_pc_pose_udp_sock >= 0) {
        close(s_pc_pose_udp_sock);
        s_pc_pose_udp_sock = -1;
    }
}

static void pc_pose_put_u16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
}

static void pc_pose_put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

static uint16_t pc_pose_get_u16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t pc_pose_get_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float pc_pose_get_f32(const uint8_t *p)
{
    const uint32_t raw = pc_pose_get_u32(p);
    float value = 0.0f;
    memcpy(&value, &raw, sizeof(value));
    return value;
}

static bool pc_pose_send_all(int sock, const uint8_t *data, size_t len)
{
    size_t sent = 0;
    while (sent < len) {
        const size_t remaining = len - sent;
        const size_t chunk = std::min(remaining, PC_POSE_TCP_SEND_CHUNK_BYTES);
        const int ret = send(sock, data + sent, chunk, 0);
        if (ret <= 0) {
            ESP_LOGW(TAG, "PC pose TCP send failed ret=%d errno=%d", ret, errno);
            return false;
        }
        sent += (size_t)ret;
        if (sent < len && PC_POSE_TCP_SEND_CHUNK_DELAY_MS > 0) {
            vTaskDelay(pdMS_TO_TICKS(PC_POSE_TCP_SEND_CHUNK_DELAY_MS));
        }
    }
    return true;
}

static size_t pc_pose_encode_rgb332(uint8_t *dst, const uint8_t *src, uint32_t width, uint32_t height)
{
    if (!dst || !src || width == 0 || height == 0) {
        return 0;
    }

    const size_t pixels = (size_t)width * (size_t)height;
    const uint16_t *src_px = reinterpret_cast<const uint16_t *>(src);
    for (size_t i = 0; i < pixels; ++i) {
        const uint16_t p = src_px[i];
        const uint8_t r3 = (uint8_t)(((p >> 11) & 0x1f) >> 2);
        const uint8_t g3 = (uint8_t)(((p >> 5) & 0x3f) >> 3);
        const uint8_t b2 = (uint8_t)((p & 0x1f) >> 3);
        dst[i] = (uint8_t)((r3 << 5) | (g3 << 2) | b2);
    }
    return pixels;
}

static bool pc_pose_encode_jpeg_hw(const uint8_t *frame,
                                   uint32_t frame_w,
                                   uint32_t frame_h,
                                   const uint8_t **payload,
                                   uint32_t *payload_len)
{
    if (!frame || !payload || !payload_len || frame_w == 0 || frame_h == 0 || s_pc_pose_jpeg_hw_disabled) {
        return false;
    }

    if (!s_pc_pose_jpeg_handle) {
        jpeg_encode_engine_cfg_t engine_cfg = {};
        engine_cfg.timeout_ms = 80;
        const esp_err_t err = jpeg_new_encoder_engine(&engine_cfg, &s_pc_pose_jpeg_handle);
        if (err != ESP_OK) {
            s_pc_pose_jpeg_hw_disabled = true;
            ESP_LOGW(TAG, "PC pose JPEG HW unavailable: %s", esp_err_to_name(err));
            return false;
        }
        ESP_LOGI(TAG, "PC pose JPEG HW encoder active");
    }

    const size_t raw_len = (size_t)frame_w * (size_t)frame_h * 2;
    const size_t out_need = raw_len + 4096;
    if (!s_pc_pose_jpeg_out || s_pc_pose_jpeg_out_cap < out_need) {
        if (s_pc_pose_jpeg_out) {
            heap_caps_free(s_pc_pose_jpeg_out);
            s_pc_pose_jpeg_out = nullptr;
            s_pc_pose_jpeg_out_cap = 0;
        }
        jpeg_encode_memory_alloc_cfg_t mem_cfg = {};
        mem_cfg.buffer_direction = JPEG_ENC_ALLOC_OUTPUT_BUFFER;
        size_t allocated = 0;
        s_pc_pose_jpeg_out = (uint8_t *)jpeg_alloc_encoder_mem(out_need, &mem_cfg, &allocated);
        if (!s_pc_pose_jpeg_out) {
            ESP_LOGW(TAG, "PC pose JPEG HW output allocation failed: %u bytes", (unsigned)out_need);
            return false;
        }
        s_pc_pose_jpeg_out_cap = allocated;
    }

    jpeg_encode_cfg_t enc_cfg = {};
    enc_cfg.height = frame_h;
    enc_cfg.width = frame_w;
    enc_cfg.src_type = JPEG_ENCODE_IN_FORMAT_RGB565;
    enc_cfg.sub_sample = JPEG_DOWN_SAMPLING_YUV420;
    enc_cfg.image_quality = PC_POSE_JPEG_QUALITY;

    uint32_t out_size = 0;
    const esp_err_t err = jpeg_encoder_process(s_pc_pose_jpeg_handle,
                                               &enc_cfg,
                                               frame,
                                               (uint32_t)raw_len,
                                               s_pc_pose_jpeg_out,
                                               (uint32_t)s_pc_pose_jpeg_out_cap,
                                               &out_size);
    if (err != ESP_OK || out_size < 4) {
        static int64_t last_warn_us = 0;
        const int64_t now_us = esp_timer_get_time();
        if (now_us - last_warn_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
            last_warn_us = now_us;
            ESP_LOGW(TAG, "PC pose JPEG HW encode failed: %s size=%u; fallback SW",
                     esp_err_to_name(err),
                     (unsigned)out_size);
        }
        return false;
    }

    const bool marker_ok = s_pc_pose_jpeg_out[0] == 0xff &&
                           s_pc_pose_jpeg_out[1] == 0xd8 &&
                           s_pc_pose_jpeg_out[out_size - 2] == 0xff &&
                           s_pc_pose_jpeg_out[out_size - 1] == 0xd9;
    if (!marker_ok) {
        static int64_t last_marker_warn_us = 0;
        const int64_t now_us = esp_timer_get_time();
        if (now_us - last_marker_warn_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
            last_marker_warn_us = now_us;
            ESP_LOGW(TAG, "PC pose JPEG HW invalid marker bytes=%u; fallback SW", (unsigned)out_size);
        }
        return false;
    }
    if (out_size > PC_POSE_MAX_JPEG_PAYLOAD_BYTES) {
        static int64_t last_size_warn_us = 0;
        const int64_t now_us = esp_timer_get_time();
        if (now_us - last_size_warn_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
            last_size_warn_us = now_us;
            ESP_LOGW(TAG, "PC pose JPEG HW too large: %u > %u; fallback SW",
                     (unsigned)out_size,
                     (unsigned)PC_POSE_MAX_JPEG_PAYLOAD_BYTES);
        }
        return false;
    }

    *payload = s_pc_pose_jpeg_out;
    *payload_len = out_size;
    return true;
}

static bool pc_pose_recv_is_hard_fail(int ret)
{
    if (ret == 0) {
        return true;
    }
    return errno != EAGAIN && errno != EWOULDBLOCK && errno != ETIMEDOUT;
}

static bool pc_pose_recv_all(int sock, uint8_t *data, size_t len, bool *connection_dead)
{
    if (connection_dead) {
        *connection_dead = false;
    }
    size_t got = 0;
    while (got < len) {
        const int ret = recv(sock, data + got, len - got, 0);
        if (ret <= 0) {
            ESP_LOGW(TAG, "PC pose TCP recv failed ret=%d errno=%d", ret, errno);
            if (connection_dead) {
                *connection_dead = pc_pose_recv_is_hard_fail(ret);
            }
            return false;
        }
        got += (size_t)ret;
    }
    return true;
}

static void pc_pose_set_socket_timeout_ms(int sock, int timeout_ms)
{
    struct timeval tv = {};
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

static bool pc_pose_connect_socket(void)
{
    if (s_pc_pose_sock >= 0) {
        return true;
    }

    const int64_t now_us = esp_timer_get_time();
    const int64_t next_us = s_pc_pose_next_connect_us.load();
    if (next_us > 0 && now_us < next_us) {
        return false;
    }

    struct sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(CONFIG_PC_POSE_SERVER_PORT);
    if (inet_pton(AF_INET, CONFIG_PC_POSE_SERVER_HOST, &dest.sin_addr) != 1) {
        ESP_LOGE(TAG, "PC pose host must be IPv4: %s", CONFIG_PC_POSE_SERVER_HOST);
        return false;
    }

    const int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "PC pose socket create failed errno=%d", errno);
        return false;
    }

    pc_pose_set_socket_timeout_ms(sock, PC_POSE_TCP_TIMEOUT_MS);
    int yes = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    int tcp_buf = PC_POSE_SOCKET_BUFFER_BYTES;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &tcp_buf, sizeof(tcp_buf));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &tcp_buf, sizeof(tcp_buf));

    ESP_LOGI(TAG, "PC pose TCP connecting to %s:%d", CONFIG_PC_POSE_SERVER_HOST, CONFIG_PC_POSE_SERVER_PORT);
    if (connect(sock, (struct sockaddr *)&dest, sizeof(dest)) != 0) {
        ESP_LOGW(TAG, "PC pose TCP connect failed errno=%d", errno);
        close(sock);
        s_pc_pose_next_connect_us.store(esp_timer_get_time() + 2000000);
        return false;
    }

    s_pc_pose_sock = sock;
    s_pc_pose_next_connect_us.store(0);
    ESP_LOGI(TAG, "PC pose TCP connected");
    return true;
}

static bool pc_pose_connect_udp_socket(void)
{
    if (s_pc_pose_udp_sock >= 0) {
        return true;
    }

    struct sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(CONFIG_PC_POSE_SERVER_PORT);
    if (inet_pton(AF_INET, CONFIG_PC_POSE_SERVER_HOST, &dest.sin_addr) != 1) {
        ESP_LOGE(TAG, "PC pose host must be IPv4: %s", CONFIG_PC_POSE_SERVER_HOST);
        return false;
    }

    const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGE(TAG, "PC pose UDP socket create failed errno=%d", errno);
        return false;
    }

    pc_pose_set_socket_timeout_ms(sock, PC_POSE_UDP_TIMEOUT_MS);
    int udp_buf = 64 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &udp_buf, sizeof(udp_buf));
    setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &udp_buf, sizeof(udp_buf));

    if (connect(sock, (struct sockaddr *)&dest, sizeof(dest)) != 0) {
        ESP_LOGW(TAG, "PC pose UDP connect failed errno=%d", errno);
        close(sock);
        return false;
    }

    s_pc_pose_udp_sock = sock;
    ESP_LOGI(TAG, "PC pose UDP ready to %s:%d", CONFIG_PC_POSE_SERVER_HOST, CONFIG_PC_POSE_SERVER_PORT);
    return true;
}

static bool pc_weather_request_once(void)
{
    if (!s_pc_pose_wifi_ready.load()) {
        set_weather_text("天气: 等待网络");
        return false;
    }

    struct sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(CONFIG_PC_POSE_SERVER_PORT);
    if (inet_pton(AF_INET, CONFIG_PC_POSE_SERVER_HOST, &dest.sin_addr) != 1) {
        ESP_LOGE(TAG, "Weather host must be IPv4: %s", CONFIG_PC_POSE_SERVER_HOST);
        set_weather_text("天气: 地址错误");
        return false;
    }

    const int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGW(TAG, "Weather socket create failed errno=%d", errno);
        return false;
    }

    pc_pose_set_socket_timeout_ms(sock, 4500);
    int yes = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

    bool ok = false;
    do {
        if (connect(sock, (struct sockaddr *)&dest, sizeof(dest)) != 0) {
            ESP_LOGW(TAG, "Weather TCP connect failed errno=%d", errno);
            break;
        }

        uint8_t req_header[PC_POSE_REQ_HEADER_BYTES] = {};
        memcpy(req_header, PC_WEATHER_REQ_MAGIC, 4);
        const uint32_t seq = ++s_weather_seq;
        pc_pose_put_u32(req_header + 12, seq);
        if (!pc_pose_send_all(sock, req_header, sizeof(req_header))) {
            break;
        }

        uint8_t resp_header[PC_WEATHER_RESP_HEADER_BYTES] = {};
        bool connection_dead = false;
        if (!pc_pose_recv_all(sock, resp_header, sizeof(resp_header), &connection_dead)) {
            break;
        }
        if (memcmp(resp_header, PC_WEATHER_RESP_MAGIC, sizeof(PC_WEATHER_RESP_MAGIC)) != 0) {
            ESP_LOGW(TAG, "Weather bad response magic");
            break;
        }
        const uint32_t payload_len = pc_pose_get_u32(resp_header + 4);
        const uint32_t resp_seq = pc_pose_get_u32(resp_header + 8);
        if (resp_seq != seq || payload_len == 0 || payload_len >= sizeof(s_weather_text)) {
            ESP_LOGW(TAG, "Weather bad response seq=%u/%u bytes=%u",
                     (unsigned)resp_seq,
                     (unsigned)seq,
                     (unsigned)payload_len);
            break;
        }

        char weather[sizeof(s_weather_text)] = {};
        if (!pc_pose_recv_all(sock, (uint8_t *)weather, payload_len, &connection_dead)) {
            break;
        }
        weather[payload_len] = '\0';
        set_weather_text(weather);
        ESP_LOGI(TAG, "Weather updated: %s", weather);
        ok = true;
    } while (false);

    close(sock);
    if (!ok) {
        set_weather_text("天气: 获取失败");
    }
    return ok;
}

static bool training_voice_fetch_pcm(VoicePromptId prompt, voice_pcm_t *out)
{
    if (!out) {
        return false;
    }
    *out = {};
    if (!s_pc_pose_wifi_ready.load()) {
        ESP_LOGW(TAG, "Voice prompt skipped: Wi-Fi not ready");
        return false;
    }

    struct sockaddr_in dest = {};
    dest.sin_family = AF_INET;
    dest.sin_port = htons(CONFIG_PC_POSE_SERVER_PORT);
    if (inet_pton(AF_INET, CONFIG_PC_POSE_SERVER_HOST, &dest.sin_addr) != 1) {
        ESP_LOGE(TAG, "Voice host must be IPv4: %s", CONFIG_PC_POSE_SERVER_HOST);
        return false;
    }

    const int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGW(TAG, "Voice socket create failed errno=%d", errno);
        return false;
    }

    pc_pose_set_socket_timeout_ms(sock, 6500);
    int yes = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

    bool ok = false;
    do {
        if (connect(sock, (struct sockaddr *)&dest, sizeof(dest)) != 0) {
            ESP_LOGW(TAG, "Voice TCP connect failed errno=%d", errno);
            break;
        }

        uint8_t req_header[PC_POSE_REQ_HEADER_BYTES] = {};
        memcpy(req_header, PC_VOICE_REQ_MAGIC, 4);
        pc_pose_put_u16(req_header + 4, (uint16_t)prompt);
        pc_pose_put_u32(req_header + 12, ++s_pc_voice_seq);
        if (!pc_pose_send_all(sock, req_header, sizeof(req_header))) {
            break;
        }

        uint8_t resp_header[PC_VOICE_RESP_HEADER_BYTES] = {};
        bool connection_dead = false;
        if (!pc_pose_recv_all(sock, resp_header, sizeof(resp_header), &connection_dead)) {
            break;
        }
        if (memcmp(resp_header, PC_VOICE_RESP_MAGIC, sizeof(PC_VOICE_RESP_MAGIC)) != 0) {
            ESP_LOGW(TAG, "Voice bad response magic");
            break;
        }

        const uint32_t sample_rate = pc_pose_get_u32(resp_header + 4);
        const uint32_t channels = pc_pose_get_u32(resp_header + 8);
        const uint32_t bits_per_sample = pc_pose_get_u32(resp_header + 12);
        const uint32_t payload_len = pc_pose_get_u32(resp_header + 16);
        if (sample_rate == 0 || channels == 0 || bits_per_sample != 16 ||
            payload_len == 0 || payload_len > PC_VOICE_MAX_PCM_BYTES) {
            ESP_LOGW(TAG,
                     "Voice bad format sr=%u ch=%u bits=%u bytes=%u",
                     (unsigned)sample_rate,
                     (unsigned)channels,
                     (unsigned)bits_per_sample,
                     (unsigned)payload_len);
            break;
        }

        uint8_t *pcm = (uint8_t *)heap_caps_malloc(payload_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!pcm) {
            pcm = (uint8_t *)heap_caps_malloc(payload_len, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        }
        if (!pcm) {
            ESP_LOGW(TAG, "Voice PCM allocation failed: %u bytes", (unsigned)payload_len);
            break;
        }

        if (!pc_pose_recv_all(sock, pcm, payload_len, &connection_dead)) {
            heap_caps_free(pcm);
            break;
        }

        out->data = pcm;
        out->len = payload_len;
        out->sample_rate = sample_rate;
        out->channels = (uint16_t)channels;
        out->bits_per_sample = (uint16_t)bits_per_sample;
        ok = true;
    } while (false);

    close(sock);
    return ok;
}

static void training_voice_release_pcm(voice_pcm_t *audio)
{
    if (!audio) {
        return;
    }
    if (audio->data) {
        heap_caps_free(audio->data);
    }
    *audio = {};
}

static void weather_task(void *arg)
{
    (void)arg;
    set_weather_text("天气: 等待网络");
    while (true) {
        if (!s_pc_pose_wifi_ready.load()) {
            set_weather_text("天气: 等待网络");
            vTaskDelay(pdMS_TO_TICKS(5000));
            continue;
        }
        const bool ok = pc_weather_request_once();
        vTaskDelay(pdMS_TO_TICKS(ok ? 10 * 60 * 1000 : 60 * 1000));
    }
}

static void start_weather_task(void)
{
    if (!s_weather_task) {
        xTaskCreatePinnedToCore(weather_task, "weather", 4096, nullptr, 1, &s_weather_task, 0);
    }
}

static int pc_pose_drain_udp_socket(void)
{
    if (s_pc_pose_udp_sock < 0 || !s_pc_pose_resp_buf || s_pc_pose_resp_cap == 0) {
        return 0;
    }
    int drained = 0;
    while (drained < 16) {
        const int ret = recv(s_pc_pose_udp_sock,
                             s_pc_pose_resp_buf,
                             s_pc_pose_resp_cap,
                             MSG_DONTWAIT);
        if (ret <= 0) {
            break;
        }
        drained++;
    }
    return drained;
}

static const char *wifi_ui_state_text(void)
{
    switch ((WifiUiState)s_wifi_ui_state.load()) {
    case WifiUiState::Connecting:
        return "连接中";
    case WifiUiState::Connected:
        return "已连接";
    case WifiUiState::Disconnected:
        return "已断开";
    case WifiUiState::Failed:
        return "连接失败";
    case WifiUiState::Idle:
    default:
        return "待连接";
    }
}

static uint32_t wifi_ui_state_color(void)
{
    switch ((WifiUiState)s_wifi_ui_state.load()) {
    case WifiUiState::Connected:
        return 0x22a06b;
    case WifiUiState::Connecting:
        return 0xd9822b;
    case WifiUiState::Failed:
        return 0xd64545;
    case WifiUiState::Disconnected:
        return 0x687783;
    case WifiUiState::Idle:
    default:
        return 0x8aa0ad;
    }
}

static void weather_lock(void)
{
    if (s_weather_mutex) {
        xSemaphoreTake(s_weather_mutex, portMAX_DELAY);
    }
}

static void weather_unlock(void)
{
    if (s_weather_mutex) {
        xSemaphoreGive(s_weather_mutex);
    }
}

static void set_weather_text(const char *text)
{
    weather_lock();
    strlcpy(s_weather_text, text ? text : "天气: --", sizeof(s_weather_text));
    weather_unlock();
}

static void get_weather_text_copy(char *out, size_t out_len)
{
    if (!out || out_len == 0) {
        return;
    }
    weather_lock();
    strlcpy(out, s_weather_text, out_len);
    weather_unlock();
}

static void start_time_sync_once(void)
{
    if (s_time_sync_started.exchange(true)) {
        return;
    }
    setenv("TZ", "CST-8", 1);
    tzset();
    esp_sntp_setoperatingmode(SNTP_OPMODE_POLL);
    esp_sntp_setservername(0, "ntp.aliyun.com");
    esp_sntp_setservername(1, "cn.pool.ntp.org");
    esp_sntp_init();
    ESP_LOGI(TAG, "SNTP time sync started");
}

static void update_datetime_ui(void)
{
    if (lbl_menu_datetime) {
        time_t now = 0;
        time(&now);
        if (now < 1700000000) {
            lv_label_set_text(lbl_menu_datetime, s_pc_pose_wifi_ready.load() ? "时间: 等待校时" : "时间: 未联网");
        } else {
            struct tm tm_now = {};
            localtime_r(&now, &tm_now);
            static const char *weekdays[] = {"周日", "周一", "周二", "周三", "周四", "周五", "周六"};
            char line[80];
            snprintf(line,
                     sizeof(line),
                     "时间: %04d-%02d-%02d %s %02d:%02d:%02d",
                     tm_now.tm_year + 1900,
                     tm_now.tm_mon + 1,
                     tm_now.tm_mday,
                     weekdays[tm_now.tm_wday],
                     tm_now.tm_hour,
                     tm_now.tm_min,
                     tm_now.tm_sec);
            lv_label_set_text(lbl_menu_datetime, line);
        }
    }
    if (lbl_menu_weather) {
        char weather[48];
        get_weather_text_copy(weather, sizeof(weather));
        lv_label_set_text(lbl_menu_weather, weather);
    }
}

static void set_wifi_ip_text(const esp_ip4_addr_t *ip)
{
    if (ip) {
        snprintf(s_wifi_ip, sizeof(s_wifi_ip), IPSTR, IP2STR(ip));
        s_wifi_ui_state.store((int)WifiUiState::Connected);
    } else {
        s_wifi_ip[0] = '\0';
    }
}

static void wifi_note_connecting(void)
{
    s_wifi_ui_state.store((int)WifiUiState::Connecting);
    s_wifi_connect_started_us.store(esp_timer_get_time());
}

static void wifi_mark_connected_ip(const esp_ip4_addr_t *ip, const char *source)
{
    if (!ip || ip->addr == 0) {
        return;
    }

    char old_ip[sizeof(s_wifi_ip)] = {};
    strlcpy(old_ip, s_wifi_ip, sizeof(old_ip));
    const bool was_ready = s_pc_pose_wifi_ready.load();

    s_pc_pose_retry = 0;
    s_pc_pose_wifi_ready.store(true);
    set_wifi_ip_text(ip);
    s_wifi_connect_started_us.store(0);
    s_wifi_last_recover_us.store(0);
    if (s_pc_pose_wifi_events) {
        xEventGroupSetBits(s_pc_pose_wifi_events, PC_POSE_WIFI_CONNECTED_BIT);
    }
    start_time_sync_once();

    if (!was_ready || strcmp(old_ip, s_wifi_ip) != 0) {
        ESP_LOGI(TAG, "Wi-Fi marked connected via %s: %s", source ? source : "unknown", s_wifi_ip);
        push_ui_msg("Wi-Fi 已连接 IP %s", s_wifi_ip);
    }
}

static bool wifi_poll_netif_ip(const char *source)
{
    if (!s_pc_pose_netif) {
        return false;
    }
    esp_netif_ip_info_t ip_info = {};
    if (esp_netif_get_ip_info(s_pc_pose_netif, &ip_info) != ESP_OK || ip_info.ip.addr == 0) {
        return false;
    }
    wifi_mark_connected_ip(&ip_info.ip, source);
    return true;
}

static void build_wifi_config(wifi_config_t *wifi_config)
{
    if (!wifi_config) {
        return;
    }
    memset(wifi_config, 0, sizeof(*wifi_config));
    strlcpy((char *)wifi_config->sta.ssid, s_wifi_ssid, sizeof(wifi_config->sta.ssid));
    strlcpy((char *)wifi_config->sta.password, s_wifi_password, sizeof(wifi_config->sta.password));
    wifi_config->sta.threshold.authmode = s_wifi_password[0] ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
}

static esp_err_t pc_pose_wifi_apply_current_config(void)
{
    if (s_wifi_ssid[0] == '\0') {
        s_wifi_ui_state.store((int)WifiUiState::Failed);
        return ESP_ERR_INVALID_ARG;
    }

    wifi_config_t wifi_config = {};
    build_wifi_config(&wifi_config);

    pc_pose_close_socket();
    s_pc_pose_wifi_ready.store(false);
    set_wifi_ip_text(nullptr);
    s_pc_pose_retry = 0;
    if (s_pc_pose_wifi_events) {
        xEventGroupClearBits(s_pc_pose_wifi_events, PC_POSE_WIFI_CONNECTED_BIT | PC_POSE_WIFI_FAIL_BIT);
    }
    wifi_note_connecting();

    esp_wifi_disconnect();
    esp_err_t err = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    if (err != ESP_OK) {
        s_wifi_ui_state.store((int)WifiUiState::Failed);
        ESP_LOGE(TAG, "esp_wifi_set_config reconnect failed: %s", esp_err_to_name(err));
        return err;
    }
    err = esp_wifi_connect();
    if (err != ESP_OK && err != ESP_ERR_WIFI_CONN) {
        s_wifi_ui_state.store((int)WifiUiState::Failed);
        ESP_LOGE(TAG, "esp_wifi_connect reconnect failed: %s", esp_err_to_name(err));
        return err;
    }
    ESP_LOGI(TAG, "PC pose Wi-Fi reconnecting to SSID:%s pass_len=%u",
             s_wifi_ssid,
             (unsigned)strlen(s_wifi_password));
    push_ui_msg("Wi-Fi 正在连接 %s", s_wifi_ssid);
    return ESP_OK;
}

static void wifi_reconnect_task(void *arg)
{
    (void)arg;
    if (!s_pc_pose_wifi_started.load()) {
        pc_pose_wifi_start();
    } else if (pc_pose_wifi_apply_current_config() != ESP_OK) {
        push_ui_msg("Wi-Fi 配置失败");
    }
    s_wifi_reconnect_task = nullptr;
    vTaskDelete(nullptr);
}

static void pc_pose_wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    (void)arg;
    if (event_base == WIFI_REMOTE_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "PC pose Wi-Fi STA start");
        wifi_note_connecting();
        esp_wifi_connect();
    } else if (event_base == WIFI_REMOTE_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "PC pose Wi-Fi disconnected");
        s_pc_pose_wifi_ready.store(false);
        set_wifi_ip_text(nullptr);
        pc_pose_close_socket();
        if (s_pc_pose_retry < CONFIG_PC_POSE_WIFI_MAX_RETRY) {
            s_pc_pose_retry++;
            wifi_note_connecting();
            ESP_LOGI(TAG, "PC pose Wi-Fi retry %d/%d", s_pc_pose_retry, CONFIG_PC_POSE_WIFI_MAX_RETRY);
            esp_wifi_connect();
        } else if (s_pc_pose_wifi_events) {
            s_wifi_ui_state.store((int)WifiUiState::Failed);
            xEventGroupSetBits(s_pc_pose_wifi_events, PC_POSE_WIFI_FAIL_BIT);
        } else {
            s_wifi_ui_state.store((int)WifiUiState::Disconnected);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "PC pose Wi-Fi got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        wifi_mark_connected_ip(&event->ip_info.ip, "event");
    }
}

static esp_err_t ensure_net_stack_ready(void)
{
    if (s_net_stack_ready) {
        return ESP_OK;
    }

    esp_err_t err = esp_netif_init();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_netif_init failed: %s", esp_err_to_name(err));
        return err;
    }

    err = esp_event_loop_create_default();
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_event_loop_create_default failed: %s", esp_err_to_name(err));
        return err;
    }

    s_net_stack_ready = true;
    ESP_LOGI(TAG, "Network stack primitives ready");
    return ESP_OK;
}

static wifi_init_config_t pc_pose_wifi_init_config(void)
{
#if defined(CONFIG_WIFI_RMT_STATIC_RX_BUFFER_NUM) && \
    defined(CONFIG_WIFI_RMT_DYNAMIC_RX_BUFFER_NUM) && \
    defined(CONFIG_WIFI_RMT_TX_BUFFER_TYPE) && \
    defined(CONFIG_WIFI_RMT_DYNAMIC_RX_MGMT_BUF) && \
    defined(CONFIG_WIFI_RMT_ESPNOW_MAX_ENCRYPT_NUM)
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    return cfg;
#else
    wifi_init_config_t cfg = {};
    cfg.osi_funcs = &g_wifi_osi_funcs;
    cfg.wpa_crypto_funcs = g_wifi_default_wpa_crypto_funcs;
    cfg.static_rx_buf_num = 10;
    cfg.dynamic_rx_buf_num = 32;
    cfg.tx_buf_type = 1;
    cfg.static_tx_buf_num = 0;
    cfg.dynamic_tx_buf_num = 32;
    cfg.rx_mgmt_buf_type = 0;
    cfg.rx_mgmt_buf_num = 5;
    cfg.cache_tx_buf_num = 0;
    cfg.csi_enable = 0;
    cfg.ampdu_rx_enable = 1;
    cfg.ampdu_tx_enable = 1;
    cfg.amsdu_tx_enable = 0;
    cfg.nvs_enable = 1;
#if CONFIG_NEWLIB_NANO_FORMAT
    cfg.nano_enable = 1;
#endif
    cfg.rx_ba_win = 6;
    cfg.wifi_task_core_id = 0;
    cfg.beacon_max_len = 752;
    cfg.mgmt_sbuf_num = 32;
    cfg.feature_caps = 0;
    cfg.sta_disconnected_pm = false;
    cfg.espnow_max_encrypt_num = 7;
    cfg.tx_hetb_queue_num = 1;
    cfg.dump_hesigb_enable = false;
    cfg.magic = WIFI_INIT_CONFIG_MAGIC;
    return cfg;
#endif
}

static bool pc_pose_wifi_start(void)
{
    if (!CONFIG_PC_POSE_ENABLE) {
        ESP_LOGI(TAG, "PC pose disabled");
        return false;
    }
    if (s_pc_pose_wifi_started.load()) {
        return s_pc_pose_wifi_ready.load();
    }
    if (s_wifi_ssid[0] == '\0') {
        ESP_LOGW(TAG, "PC pose Wi-Fi SSID is empty");
        s_wifi_ui_state.store((int)WifiUiState::Failed);
        return false;
    }

    esp_err_t err = ensure_net_stack_ready();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "PC pose network primitives unavailable: %s", esp_err_to_name(err));
        s_wifi_ui_state.store((int)WifiUiState::Failed);
        return false;
    }

    if (!s_pc_pose_wifi_events) {
        s_pc_pose_wifi_events = xEventGroupCreate();
        if (!s_pc_pose_wifi_events) {
            ESP_LOGE(TAG, "PC pose Wi-Fi event group allocation failed");
            return false;
        }
    }
    if (!s_pc_pose_netif) {
        s_pc_pose_netif = esp_netif_create_default_wifi_sta();
        if (!s_pc_pose_netif) {
            ESP_LOGE(TAG, "esp_netif_create_default_wifi_sta failed");
            return false;
        }
        ESP_LOGI(TAG, "PC pose STA netif created");
    }

    wifi_init_config_t cfg = pc_pose_wifi_init_config();
    err = esp_wifi_init(&cfg);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "esp_wifi_init failed: %s", esp_err_to_name(err));
        return false;
    }

    err = esp_event_handler_register(WIFI_REMOTE_EVENT, ESP_EVENT_ANY_ID, pc_pose_wifi_event_handler, nullptr);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "wifi event handler register failed: %s", esp_err_to_name(err));
        return false;
    }
    err = esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, pc_pose_wifi_event_handler, nullptr);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
        ESP_LOGE(TAG, "ip event handler register failed: %s", esp_err_to_name(err));
        return false;
    }

    wifi_config_t wifi_config = {};
    build_wifi_config(&wifi_config);

    err = esp_wifi_set_mode(WIFI_MODE_STA);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_mode failed: %s", esp_err_to_name(err));
        return false;
    }
    err = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_set_config failed: %s", esp_err_to_name(err));
        s_wifi_ui_state.store((int)WifiUiState::Failed);
        return false;
    }
    s_wifi_ui_state.store((int)WifiUiState::Connecting);
    err = esp_wifi_start();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_wifi_start failed: %s", esp_err_to_name(err));
        s_wifi_ui_state.store((int)WifiUiState::Failed);
        return false;
    }
    s_pc_pose_wifi_started.store(true);

    ESP_LOGI(TAG, "PC pose Wi-Fi connecting to SSID:%s pass_len=%u",
             s_wifi_ssid,
             (unsigned)strlen(s_wifi_password));
    EventBits_t bits = xEventGroupWaitBits(s_pc_pose_wifi_events,
                                           PC_POSE_WIFI_CONNECTED_BIT | PC_POSE_WIFI_FAIL_BIT,
                                           pdFALSE,
                                           pdFALSE,
                                           pdMS_TO_TICKS(CONFIG_PC_POSE_WIFI_CONNECT_TIMEOUT_MS));
    if (bits & PC_POSE_WIFI_CONNECTED_BIT) {
        push_ui_msg("PC pose Wi-Fi ready");
        return true;
    }
    if (bits & PC_POSE_WIFI_FAIL_BIT) {
        push_ui_msg("PC pose Wi-Fi failed");
        s_wifi_ui_state.store((int)WifiUiState::Failed);
        return false;
    }
    if (wifi_poll_netif_ip("start-timeout")) {
        push_ui_msg("PC pose Wi-Fi ready");
        return true;
    }

    push_ui_msg("PC pose Wi-Fi connecting...");
    return false;
}

static void pc_pose_wifi_task(void *arg)
{
    (void)arg;
    vTaskDelay(pdMS_TO_TICKS(2500));
    ESP_LOGI(TAG, "PC pose Wi-Fi task start");
    pc_pose_wifi_start();
    vTaskDelete(nullptr);
}

static void wifi_watchdog_task(void *arg)
{
    (void)arg;
    vTaskDelay(pdMS_TO_TICKS(6000));
    while (true) {
        if (!CONFIG_PC_POSE_ENABLE || !s_pc_pose_wifi_started.load()) {
            vTaskDelay(pdMS_TO_TICKS(3000));
            continue;
        }

        if (!s_pc_pose_wifi_ready.load() &&
            (WifiUiState)s_wifi_ui_state.load() == WifiUiState::Connecting) {
            if (wifi_poll_netif_ip("watchdog")) {
                vTaskDelay(pdMS_TO_TICKS(3000));
                continue;
            }

            const int64_t now_us = esp_timer_get_time();
            const int64_t started_us = s_wifi_connect_started_us.load();
            const int64_t last_recover_us = s_wifi_last_recover_us.load();
            const bool stuck_connecting =
                started_us > 0 &&
                now_us - started_us >= 15000000 &&
                (last_recover_us == 0 || now_us - last_recover_us >= 15000000);
            if (stuck_connecting) {
                s_wifi_last_recover_us.store(now_us);
                ESP_LOGW(TAG, "Wi-Fi watchdog reconnect: state stuck in connecting");
                push_ui_msg("Wi-Fi 连接超时，自动重试");
                pc_pose_close_socket();
                esp_wifi_disconnect();
                vTaskDelay(pdMS_TO_TICKS(250));
                wifi_note_connecting();
                esp_err_t err = esp_wifi_connect();
                if (err != ESP_OK && err != ESP_ERR_WIFI_CONN) {
                    ESP_LOGW(TAG, "Wi-Fi watchdog connect failed: %s", esp_err_to_name(err));
                    s_wifi_ui_state.store((int)WifiUiState::Failed);
                }
            }
        } else {
            s_wifi_last_recover_us.store(0);
        }

        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}

static bool parse_pc_pose_binary_response(const uint8_t *payload,
                                          size_t payload_len,
                                          uint16_t people_count,
                                          uint16_t kpt_count,
                                          uint32_t dst_w,
                                          uint32_t dst_h,
                                          std::vector<pose_result_t> *out,
                                          int *valid_count,
                                          bool trust_pc_header)
{
    if (!payload || !out || dst_w == 0 || dst_h == 0 || kpt_count == 0) {
        return false;
    }
    const size_t floats_per_person = (size_t)kpt_count * PC_POSE_FLOATS_PER_KPT;
    const size_t expected_len = (size_t)people_count * floats_per_person * sizeof(float);
    if (payload_len != expected_len) {
        ESP_LOGW(TAG, "PC pose binary payload size mismatch: got=%u expected=%u",
                 (unsigned)payload_len,
                 (unsigned)expected_len);
        return false;
    }

    auto decode_variant = [&](bool swap_xy, std::vector<pose_result_t> *decoded, int *decoded_valid) {
        decoded->clear();
        int total_valid = 0;
        const uint16_t people_to_parse = std::min<uint16_t>(people_count, POSE_MAX_PEOPLE);
        const uint16_t kpts_to_parse = std::min<uint16_t>(kpt_count, POSE_KPTS);
        (void)trust_pc_header;
        const float score_thr = PC_POSE_PARSE_SCORE_THR;
        const int min_visible = PC_POSE_MIN_VISIBLE_KPTS;
        const int min_box_px = std::max(8, (int)lroundf((float)std::min(dst_w, dst_h) * PC_POSE_MIN_BOX_RATIO));
        const uint8_t *p = payload;

        for (uint16_t person_idx = 0; person_idx < people_count; person_idx++) {
            pose_result_t pose = {};
            pose.score = 0.0f;
            pose.x1 = (int)dst_w - 1;
            pose.y1 = (int)dst_h - 1;
            pose.x2 = 0;
            pose.y2 = 0;
            pose.keypoints.reserve(POSE_KPTS);

            int visible = 0;
            float score_sum = 0.0f;
            for (uint16_t k = 0; k < kpt_count; k++) {
                const float x = pc_pose_get_f32(p + 0);
                const float y = pc_pose_get_f32(p + 4);
                const float score = pc_pose_get_f32(p + 12);
                p += PC_POSE_FLOATS_PER_KPT * sizeof(float);

                if (person_idx >= people_to_parse || k >= kpts_to_parse) {
                    continue;
                }

                const float rx = swap_xy ? y : x;
                const float ry = swap_xy ? x : y;
                pose_keypoint_t kp = {};
                kp.x = std::max(0, std::min((int)dst_w - 1, (int)lroundf(rx * (float)(dst_w - 1))));
                kp.y = std::max(0, std::min((int)dst_h - 1, (int)lroundf(ry * (float)(dst_h - 1))));
                kp.score = score;
                pose.keypoints.push_back(kp);

                score_sum += score;
                if (score >= score_thr) {
                    pose.x1 = std::min(pose.x1, kp.x);
                    pose.y1 = std::min(pose.y1, kp.y);
                    pose.x2 = std::max(pose.x2, kp.x);
                    pose.y2 = std::max(pose.y2, kp.y);
                    visible++;
                }
            }

            if (person_idx >= people_to_parse) {
                continue;
            }
            while ((int)pose.keypoints.size() < POSE_KPTS) {
                pose.keypoints.push_back({});
            }
            total_valid += visible;
            pose.score = kpts_to_parse > 0 ? score_sum / (float)kpts_to_parse : 0.0f;
            const bool box_ok = pose.x2 > pose.x1 && pose.y2 > pose.y1 &&
                                (pose.x2 - pose.x1) >= min_box_px && (pose.y2 - pose.y1) >= min_box_px;
            if (visible >= min_visible && box_ok) {
                decoded->push_back(pose);
            }
        }

        if (decoded_valid) {
            *decoded_valid = total_valid;
        }
        return !decoded->empty();
    };

    std::vector<pose_result_t> normal;
    int normal_valid = 0;
    const bool normal_ok = decode_variant(false, &normal, &normal_valid);
    *out = normal;
    if (valid_count) {
        *valid_count = normal_valid;
    }
    static int64_t last_decode_log_us = 0;
    const int64_t now_us = esp_timer_get_time();
    if (now_us - last_decode_log_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
        last_decode_log_us = now_us;
        ESP_LOGI(TAG, "PC pose decode choose=%s valid=%d out=%d",
                 "normal_xy",
                 normal_valid,
                 (int)out->size());
    }
    return normal_ok;
}

static bool run_pc_pose_request(const uint8_t *frame,
                                uint32_t frame_w,
                                uint32_t frame_h,
                                std::vector<pose_result_t> *results,
                                float *http_ms,
                                float *tx_ms,
                                float *wait_ms,
                                float *infer_ms,
                                int *valid_count)
{
    if (!CONFIG_PC_POSE_ENABLE || !frame || !results || frame_w == 0 || frame_h == 0) {
        return false;
    }
    if (!s_pc_pose_wifi_ready.load()) {
        return false;
    }

    if (!s_pc_pose_resp_buf) {
        s_pc_pose_resp_cap = CONFIG_PC_POSE_RESPONSE_MAX_BYTES;
        s_pc_pose_resp_buf = (uint8_t *)heap_caps_malloc(s_pc_pose_resp_cap, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!s_pc_pose_resp_buf) {
            s_pc_pose_resp_buf = (uint8_t *)heap_caps_malloc(s_pc_pose_resp_cap, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        }
        if (!s_pc_pose_resp_buf) {
            ESP_LOGE(TAG, "PC pose response buffer allocation failed");
            return false;
        }
    }

    const size_t rgb332_payload_cap = (size_t)frame_w * (size_t)frame_h;
    if (PC_POSE_USE_RGB332 && (!s_pc_pose_tx_buf || s_pc_pose_tx_cap < rgb332_payload_cap)) {
        if (s_pc_pose_tx_buf) {
            heap_caps_free(s_pc_pose_tx_buf);
            s_pc_pose_tx_buf = nullptr;
            s_pc_pose_tx_cap = 0;
        }
        s_pc_pose_tx_buf = (uint8_t *)heap_caps_malloc(rgb332_payload_cap, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (!s_pc_pose_tx_buf) {
            s_pc_pose_tx_buf = (uint8_t *)heap_caps_malloc(rgb332_payload_cap, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        }
        if (!s_pc_pose_tx_buf) {
            ESP_LOGE(TAG, "PC pose tx buffer allocation failed: %u bytes", (unsigned)rgb332_payload_cap);
            return false;
        }
        s_pc_pose_tx_cap = rgb332_payload_cap;
        ESP_LOGI(TAG, "PC pose tx format: RGB332 %ux%u payload=%u bytes",
                 (unsigned)frame_w,
                 (unsigned)frame_h,
                 (unsigned)rgb332_payload_cap);
    }

    dl::image::jpeg_img_t jpeg_img = {};
    const uint8_t *payload = frame;
    uint32_t payload_len = (uint32_t)((size_t)frame_w * (size_t)frame_h * 2);
    const uint8_t *req_magic = PC_POSE_REQ_MAGIC_RGB565;
    const char *tx_format = "RGB565";

    if (PC_POSE_USE_JPEG) {
        const uint8_t *hw_jpeg_payload = nullptr;
        uint32_t hw_jpeg_len = 0;
        if (pc_pose_encode_jpeg_hw(frame, frame_w, frame_h, &hw_jpeg_payload, &hw_jpeg_len) &&
            hw_jpeg_len > 0 && hw_jpeg_len < payload_len) {
            payload = hw_jpeg_payload;
            payload_len = hw_jpeg_len;
            req_magic = PC_POSE_REQ_MAGIC_JPEG;
            tx_format = "JPEG-HW";
        } else {
            dl::image::img_t img = {};
            img.data = (void *)frame;
            img.width = (uint16_t)frame_w;
            img.height = (uint16_t)frame_h;
            img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565LE;
            jpeg_img = dl::image::sw_encode_jpeg(img, PC_POSE_JPEG_QUALITY);
            if (jpeg_img.data && jpeg_img.data_len > 0) {
                const uint8_t *jpg = (const uint8_t *)jpeg_img.data;
                const bool jpeg_markers_ok = jpeg_img.data_len >= 4 &&
                                             jpg[0] == 0xff &&
                                             jpg[1] == 0xd8 &&
                                             jpg[jpeg_img.data_len - 2] == 0xff &&
                                             jpg[jpeg_img.data_len - 1] == 0xd9;
                if (!jpeg_markers_ok) {
                    ESP_LOGW(TAG, "PC pose JPEG invalid marker, skip frame bytes=%u", (unsigned)jpeg_img.data_len);
                    heap_caps_free(jpeg_img.data);
                    return false;
                }
                if (jpeg_img.data_len > PC_POSE_MAX_JPEG_PAYLOAD_BYTES) {
                    ESP_LOGW(TAG, "PC pose JPEG-SW too large: %u > %u, skip frame",
                             (unsigned)jpeg_img.data_len,
                             (unsigned)PC_POSE_MAX_JPEG_PAYLOAD_BYTES);
                    heap_caps_free(jpeg_img.data);
                    return false;
                }
                if (jpeg_img.data_len < payload_len) {
                    payload = jpg;
                    payload_len = (uint32_t)jpeg_img.data_len;
                    req_magic = PC_POSE_REQ_MAGIC_JPEG;
                    tx_format = "JPEG-SW";
                } else {
                    heap_caps_free(jpeg_img.data);
                    jpeg_img = {};
                    ESP_LOGW(TAG, "PC pose JPEG larger than RGB565, skip raw frame");
                }
            } else {
                if (jpeg_img.data) {
                    heap_caps_free(jpeg_img.data);
                    jpeg_img = {};
                }
                ESP_LOGW(TAG, "PC pose JPEG encode skipped/failed");
            }
        }
    }
    if (PC_POSE_USE_JPEG && req_magic != PC_POSE_REQ_MAGIC_JPEG) {
        if (jpeg_img.data) {
            heap_caps_free(jpeg_img.data);
        }
        ESP_LOGW(TAG, "PC pose JPEG unavailable, skip raw frame");
        return false;
    }
    if (PC_POSE_USE_RGB332 && req_magic == PC_POSE_REQ_MAGIC_RGB565) {
        payload = s_pc_pose_tx_buf;
        payload_len = (uint32_t)pc_pose_encode_rgb332(s_pc_pose_tx_buf, frame, frame_w, frame_h);
        req_magic = PC_POSE_REQ_MAGIC_RGB332;
        tx_format = "RGB332";
    }

    if (payload_len == 0 || !payload) {
        if (jpeg_img.data) {
            heap_caps_free(jpeg_img.data);
        }
        ESP_LOGW(TAG, "PC pose empty payload");
        return false;
    }

    uint8_t req_header[PC_POSE_REQ_HEADER_BYTES] = {};
    memcpy(req_header, req_magic, 4);
    pc_pose_put_u16(req_header + 4, (uint16_t)frame_w);
    pc_pose_put_u16(req_header + 6, (uint16_t)frame_h);
    pc_pose_put_u32(req_header + 8, payload_len);
    const uint32_t frame_seq = ++s_pc_pose_frame_seq;
    pc_pose_put_u32(req_header + 12, frame_seq);

    static const char *last_tx_format = nullptr;
    if (last_tx_format != tx_format) {
        ESP_LOGI(TAG,
                 "PC pose tx format: %s%s %ux%u payload=%u bytes",
                 PC_POSE_USE_UDP ? "UDP " : "",
                 tx_format,
                 (unsigned)frame_w,
                 (unsigned)frame_h,
                 (unsigned)payload_len);
        last_tx_format = tx_format;
    }

    const int64_t udp_try_us = esp_timer_get_time();
    static int udp_miss_count = 0;
    if (PC_POSE_USE_UDP && udp_try_us >= s_pc_pose_udp_retry_after_us.load() &&
        sizeof(req_header) + payload_len <= PC_POSE_MAX_UDP_PACKET_BYTES) {
        const size_t packet_len = sizeof(req_header) + payload_len;
        if (!s_pc_pose_packet_buf || s_pc_pose_packet_cap < packet_len) {
            if (s_pc_pose_packet_buf) {
                heap_caps_free(s_pc_pose_packet_buf);
                s_pc_pose_packet_buf = nullptr;
                s_pc_pose_packet_cap = 0;
            }
            s_pc_pose_packet_buf = (uint8_t *)heap_caps_malloc(packet_len, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
            if (!s_pc_pose_packet_buf) {
                s_pc_pose_packet_buf = (uint8_t *)heap_caps_malloc(packet_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
            }
            if (!s_pc_pose_packet_buf) {
                ESP_LOGE(TAG, "PC pose UDP packet buffer allocation failed: %u bytes", (unsigned)packet_len);
                if (jpeg_img.data) {
                    heap_caps_free(jpeg_img.data);
                }
                return false;
            }
            s_pc_pose_packet_cap = packet_len;
        }
        memcpy(s_pc_pose_packet_buf, req_header, sizeof(req_header));
        memcpy(s_pc_pose_packet_buf + sizeof(req_header), payload, payload_len);

        if (!pc_pose_connect_udp_socket()) {
            ESP_LOGW(TAG, "PC pose UDP unavailable, fallback TCP");
            s_pc_pose_udp_retry_after_us.store(esp_timer_get_time() + PC_POSE_UDP_FAIL_BACKOFF_US);
        } else {
            const int drained = pc_pose_drain_udp_socket();
            if (drained > 0) {
                static int64_t last_udp_drain_log_us = 0;
                const int64_t drain_log_us = esp_timer_get_time();
                if (drain_log_us - last_udp_drain_log_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
                    last_udp_drain_log_us = drain_log_us;
                    ESP_LOGW(TAG, "PC pose UDP drained %d stale packet(s) before send", drained);
                }
            }

            const int64_t start_us = esp_timer_get_time();
            const ssize_t sent = send(s_pc_pose_udp_sock, s_pc_pose_packet_buf, packet_len, 0);
            const int64_t tx_end_us = esp_timer_get_time();
            if (tx_ms) {
                *tx_ms = (tx_end_us - start_us) / 1000.0f;
            }
            if (sent != (ssize_t)packet_len) {
                udp_miss_count++;
                const bool reset_udp = udp_miss_count >= PC_POSE_UDP_MISSES_BEFORE_TCP_FALLBACK;
                ESP_LOGW(TAG, "PC pose UDP send failed sent=%d len=%u errno=%d miss=%d/%d%s",
                         (int)sent,
                         (unsigned)packet_len,
                         errno,
                         udp_miss_count,
                         PC_POSE_UDP_MISSES_BEFORE_TCP_FALLBACK,
                         reset_udp ? "; reset UDP" : "; skip TCP");
                close(s_pc_pose_udp_sock);
                s_pc_pose_udp_sock = -1;
                s_pc_pose_udp_retry_after_us.store(esp_timer_get_time() + PC_POSE_UDP_RESET_BACKOFF_US);
                if (reset_udp) {
                    udp_miss_count = 0;
                }
                if (jpeg_img.data) {
                    heap_caps_free(jpeg_img.data);
                }
                return false;
            } else {
                int stale_count = 0;
                while (true) {
                    const ssize_t rx_len = recv(s_pc_pose_udp_sock, s_pc_pose_resp_buf, s_pc_pose_resp_cap, 0);
                    const int64_t end_us = esp_timer_get_time();
                    if (wait_ms) {
                        *wait_ms = (end_us - tx_end_us) / 1000.0f;
                    }
                    if (http_ms) {
                        *http_ms = (end_us - start_us) / 1000.0f;
                    }
                    if (rx_len < (ssize_t)PC_POSE_RESP_HEADER_BYTES) {
                        udp_miss_count++;
                        const bool reset_udp = udp_miss_count >= PC_POSE_UDP_MISSES_BEFORE_TCP_FALLBACK;
                        ESP_LOGW(TAG, "PC pose UDP response failed len=%d errno=%d stale=%d miss=%d/%d%s",
                                 (int)rx_len,
                                 errno,
                                 stale_count,
                                 udp_miss_count,
                                 PC_POSE_UDP_MISSES_BEFORE_TCP_FALLBACK,
                                 reset_udp ? "; reset UDP" : "; keep UDP");
                        if (reset_udp) {
                            close(s_pc_pose_udp_sock);
                            s_pc_pose_udp_sock = -1;
                            s_pc_pose_udp_retry_after_us.store(esp_timer_get_time() + PC_POSE_UDP_RESET_BACKOFF_US);
                            udp_miss_count = 0;
                        }
                        if (jpeg_img.data) {
                            heap_caps_free(jpeg_img.data);
                        }
                        return false;
                    }
                    const uint8_t *resp_header = s_pc_pose_resp_buf;
                    if (memcmp(resp_header, PC_POSE_RESP_MAGIC, sizeof(PC_POSE_RESP_MAGIC)) != 0) {
                        udp_miss_count++;
                        ESP_LOGW(TAG, "PC pose UDP bad response magic miss=%d/%d; reset UDP",
                                 udp_miss_count,
                                 PC_POSE_UDP_MISSES_BEFORE_TCP_FALLBACK);
                        close(s_pc_pose_udp_sock);
                        s_pc_pose_udp_sock = -1;
                        s_pc_pose_udp_retry_after_us.store(esp_timer_get_time() + PC_POSE_UDP_RESET_BACKOFF_US);
                        if (jpeg_img.data) {
                            heap_caps_free(jpeg_img.data);
                        }
                        return false;
                    }

                    const uint16_t people = pc_pose_get_u16(resp_header + 4);
                    const uint16_t kpts = pc_pose_get_u16(resp_header + 6);
                    const uint32_t valid = pc_pose_get_u32(resp_header + 8);
                    const float infer = pc_pose_get_f32(resp_header + 12);
                    const uint32_t resp_payload_len = pc_pose_get_u32(resp_header + 16);
                    const uint32_t resp_seq = pc_pose_get_u32(resp_header + 20);
                    const size_t got_payload_len = (size_t)rx_len - PC_POSE_RESP_HEADER_BYTES;
                    if (resp_seq != frame_seq) {
                        stale_count++;
                        if (stale_count <= 2) {
                            static int64_t last_udp_stale_log_us = 0;
                            const int64_t stale_log_us = esp_timer_get_time();
                            if (stale_log_us - last_udp_stale_log_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
                                last_udp_stale_log_us = stale_log_us;
                                ESP_LOGW(TAG, "PC pose UDP stale response seq=%u want=%u; keep reading current",
                                     (unsigned)resp_seq,
                                     (unsigned)frame_seq);
                            }
                        }
                        continue;
                    }

                    if (infer_ms) {
                        *infer_ms = infer;
                    }
                    if (valid_count) {
                        *valid_count = (int)valid;
                    }
                    if (resp_payload_len == got_payload_len) {
                        const bool parsed = parse_pc_pose_binary_response(s_pc_pose_resp_buf + PC_POSE_RESP_HEADER_BYTES,
                                                                           resp_payload_len,
                                                                           people,
                                                                           kpts,
                                                                           frame_w,
                                                                           frame_h,
                                                                           results,
                                                                           valid_count,
                                                                           people > 0);
                        if (jpeg_img.data) {
                            heap_caps_free(jpeg_img.data);
                            jpeg_img = {};
                        }
                        s_pc_pose_udp_retry_after_us.store(0);
                        udp_miss_count = 0;
                        return parsed;
                    }

                    udp_miss_count++;
                    ESP_LOGW(TAG, "PC pose UDP payload size mismatch: header=%u got=%u miss=%d/%d; reset UDP",
                             (unsigned)resp_payload_len,
                             (unsigned)got_payload_len,
                             udp_miss_count,
                             PC_POSE_UDP_MISSES_BEFORE_TCP_FALLBACK);
                    close(s_pc_pose_udp_sock);
                    s_pc_pose_udp_sock = -1;
                    s_pc_pose_udp_retry_after_us.store(esp_timer_get_time() + PC_POSE_UDP_RESET_BACKOFF_US);
                    if (jpeg_img.data) {
                        heap_caps_free(jpeg_img.data);
                    }
                    return false;
                }
            }
        }
    }

    if (!pc_pose_connect_socket()) {
        if (jpeg_img.data) {
            heap_caps_free(jpeg_img.data);
        }
        return false;
    }

    uint8_t resp_header[PC_POSE_RESP_HEADER_BYTES] = {};
    const int64_t start_us = esp_timer_get_time();
    bool ok = pc_pose_send_all(s_pc_pose_sock, req_header, sizeof(req_header)) &&
              pc_pose_send_all(s_pc_pose_sock, payload, payload_len);
    if (jpeg_img.data) {
        heap_caps_free(jpeg_img.data);
        jpeg_img = {};
    }
    const int64_t tx_end_us = esp_timer_get_time();
    if (tx_ms) {
        *tx_ms = (tx_end_us - start_us) / 1000.0f;
    }
    bool recv_dead = false;
    if (ok) {
        ok = pc_pose_recv_all(s_pc_pose_sock, resp_header, sizeof(resp_header), &recv_dead);
    }
    const int64_t header_end_us = esp_timer_get_time();
    if (wait_ms) {
        *wait_ms = (header_end_us - tx_end_us) / 1000.0f;
    }
    if (!ok) {
        if (http_ms) {
            *http_ms = (header_end_us - start_us) / 1000.0f;
        }
        pc_pose_close_socket();
        s_pc_pose_next_connect_us.store(esp_timer_get_time() + PC_POSE_TCP_FAIL_BACKOFF_US);
        return false;
    }

    if (memcmp(resp_header, PC_POSE_RESP_MAGIC, sizeof(PC_POSE_RESP_MAGIC)) != 0) {
        ESP_LOGW(TAG, "PC pose bad response magic");
        pc_pose_close_socket();
        return false;
    }

    const uint16_t people = pc_pose_get_u16(resp_header + 4);
    const uint16_t kpts = pc_pose_get_u16(resp_header + 6);
    const uint32_t valid = pc_pose_get_u32(resp_header + 8);
    const float infer = pc_pose_get_f32(resp_header + 12);
    const uint32_t resp_payload_len = pc_pose_get_u32(resp_header + 16);
    const uint32_t resp_seq = pc_pose_get_u32(resp_header + 20);
    if (resp_seq != frame_seq) {
        ESP_LOGW(TAG, "PC pose TCP stale response seq=%u want=%u",
                 (unsigned)resp_seq,
                 (unsigned)frame_seq);
        pc_pose_close_socket();
        return false;
    }
    if (infer_ms) {
        *infer_ms = infer;
    }
    if (valid_count) {
        *valid_count = (int)valid;
    }
    if (resp_payload_len > s_pc_pose_resp_cap) {
        ESP_LOGW(TAG, "PC pose response too large: %u > %u", (unsigned)resp_payload_len, (unsigned)s_pc_pose_resp_cap);
        pc_pose_close_socket();
        return false;
    }
    recv_dead = false;
    if (resp_payload_len > 0 && !pc_pose_recv_all(s_pc_pose_sock, s_pc_pose_resp_buf, resp_payload_len, &recv_dead)) {
        if (http_ms) {
            *http_ms = (esp_timer_get_time() - start_us) / 1000.0f;
        }
        pc_pose_close_socket();
        return false;
    }
    const int64_t end_us = esp_timer_get_time();
    if (http_ms) {
        *http_ms = (end_us - start_us) / 1000.0f;
    }

    return parse_pc_pose_binary_response(s_pc_pose_resp_buf,
                                         resp_payload_len,
                                         people,
                                         kpts,
                                         frame_w,
                                         frame_h,
                                         results,
                                         valid_count,
                                         people > 0);
}

static int scale_coord(int v, uint32_t src, uint32_t dst)
{
    if (src == 0) {
        return 0;
    }
    return (int)(((int64_t)v * (int64_t)dst + (int64_t)src / 2) / (int64_t)src);
}

static int scale_pose_coord_centered(int v, uint32_t src, uint32_t dst)
{
    if (src == 0 || dst == 0) {
        return 0;
    }
    const float mapped = (((float)v + 0.5f) * (float)dst / (float)src) - 0.5f;
    return (int)lroundf(mapped);
}

static std::list<dl::detect::result_t> scale_detections(const std::list<dl::detect::result_t> &src,
                                                        uint32_t src_w,
                                                        uint32_t src_h,
                                                        uint32_t dst_w,
                                                        uint32_t dst_h,
                                                        bool rotate_180 = false)
{
    std::list<dl::detect::result_t> out;
    for (auto det : src) {
        if (det.box.size() >= 4) {
            int x1 = scale_coord(det.box[0], src_w, dst_w);
            int y1 = scale_coord(det.box[1], src_h, dst_h);
            int x2 = scale_coord(det.box[2], src_w, dst_w);
            int y2 = scale_coord(det.box[3], src_h, dst_h);
            if (rotate_180) {
                det.box[0] = (int)dst_w - 1 - x2;
                det.box[1] = (int)dst_h - 1 - y2;
                det.box[2] = (int)dst_w - 1 - x1;
                det.box[3] = (int)dst_h - 1 - y1;
            } else {
                det.box[0] = x1;
                det.box[1] = y1;
                det.box[2] = x2;
                det.box[3] = y2;
            }
            det.limit_box((int)dst_w, (int)dst_h);
        }
        for (size_t i = 0; i + 1 < det.keypoint.size(); i += 2) {
            int x = scale_coord(det.keypoint[i], src_w, dst_w);
            int y = scale_coord(det.keypoint[i + 1], src_h, dst_h);
            if (rotate_180) {
                x = (int)dst_w - 1 - x;
                y = (int)dst_h - 1 - y;
            }
            det.keypoint[i] = x;
            det.keypoint[i + 1] = y;
        }
        if (!det.keypoint.empty()) {
            det.limit_keypoint((int)dst_w, (int)dst_h);
        }
        out.push_back(det);
    }
    return out;
}

static std::vector<pose_result_t> scale_pose_results(const std::vector<pose_result_t> &src,
                                                     uint32_t src_w,
                                                     uint32_t src_h,
                                                     uint32_t dst_w,
                                                     uint32_t dst_h,
                                                     bool rotate_180 = false)
{
    std::vector<pose_result_t> out;
    out.reserve(src.size());
    for (auto pose : src) {
        int x1 = scale_pose_coord_centered(pose.x1, src_w, dst_w);
        int y1 = scale_pose_coord_centered(pose.y1, src_h, dst_h);
        int x2 = scale_pose_coord_centered(pose.x2, src_w, dst_w);
        int y2 = scale_pose_coord_centered(pose.y2, src_h, dst_h);
        if (rotate_180) {
            pose.x1 = (int)dst_w - 1 - x2;
            pose.y1 = (int)dst_h - 1 - y2;
            pose.x2 = (int)dst_w - 1 - x1;
            pose.y2 = (int)dst_h - 1 - y1;
        } else {
            pose.x1 = x1;
            pose.y1 = y1;
            pose.x2 = x2;
            pose.y2 = y2;
        }
        for (auto &pt : pose.keypoints) {
            int x = scale_pose_coord_centered(pt.x, src_w, dst_w);
            int y = scale_pose_coord_centered(pt.y, src_h, dst_h);
            if (rotate_180) {
                x = (int)dst_w - 1 - x;
                y = (int)dst_h - 1 - y;
            }
            pt.x = x;
            pt.y = y;
        }
        out.push_back(pose);
    }
    return out;
}

static std::vector<pose_result_t> scale_pose_results_from_crop(const std::vector<pose_result_t> &src,
                                                               uint32_t src_w,
                                                               uint32_t src_h,
                                                               uint32_t crop_x,
                                                               uint32_t crop_y,
                                                               uint32_t crop_w,
                                                               uint32_t crop_h,
                                                               uint32_t dst_w,
                                                               uint32_t dst_h,
                                                               bool rotate_180 = false)
{
    std::vector<pose_result_t> out;
    out.reserve(src.size());
    for (auto pose : src) {
        auto map_x = [&](int x) {
            int mapped = scale_pose_coord_centered(x, src_w, crop_w);
            return rotate_180 ? (int)(crop_x + crop_w - 1) - mapped : (int)crop_x + mapped;
        };
        auto map_y = [&](int y) {
            int mapped = scale_pose_coord_centered(y, src_h, crop_h);
            return rotate_180 ? (int)(crop_y + crop_h - 1) - mapped : (int)crop_y + mapped;
        };

        int x1 = map_x(pose.x1);
        int y1 = map_y(pose.y1);
        int x2 = map_x(pose.x2);
        int y2 = map_y(pose.y2);
        pose.x1 = std::max(0, std::min((int)dst_w - 1, std::min(x1, x2)));
        pose.y1 = std::max(0, std::min((int)dst_h - 1, std::min(y1, y2)));
        pose.x2 = std::max(0, std::min((int)dst_w - 1, std::max(x1, x2)));
        pose.y2 = std::max(0, std::min((int)dst_h - 1, std::max(y1, y2)));

        for (auto &pt : pose.keypoints) {
            pt.x = std::max(0, std::min((int)dst_w - 1, map_x(pt.x)));
            pt.y = std::max(0, std::min((int)dst_h - 1, map_y(pt.y)));
        }
        out.push_back(pose);
    }
    return out;
}

static std::vector<pose_result_t> scale_pose_results_from_letterbox(const std::vector<pose_result_t> &src,
                                                                    uint32_t src_w,
                                                                    uint32_t src_h,
                                                                    uint32_t box_x,
                                                                    uint32_t box_y,
                                                                    uint32_t box_w,
                                                                    uint32_t box_h,
                                                                    uint32_t dst_w,
                                                                    uint32_t dst_h,
                                                                    bool rotate_180 = false)
{
    std::vector<pose_result_t> out;
    if (src_w == 0 || src_h == 0 || box_w == 0 || box_h == 0 || dst_w == 0 || dst_h == 0) {
        return out;
    }

    box_x = std::min(box_x, src_w - 1);
    box_y = std::min(box_y, src_h - 1);
    box_w = std::min(box_w, src_w - box_x);
    box_h = std::min(box_h, src_h - box_y);

    out.reserve(src.size());
    for (auto pose : src) {
        auto map_x = [&](int x) {
            const int min_x = (int)box_x;
            const int max_x = (int)(box_x + box_w - 1);
            const int clamped = std::max(min_x, std::min(max_x, x));
            const int rel = clamped - min_x;
            int mapped = scale_pose_coord_centered(rel, box_w, dst_w);
            return rotate_180 ? (int)dst_w - 1 - mapped : mapped;
        };
        auto map_y = [&](int y) {
            const int min_y = (int)box_y;
            const int max_y = (int)(box_y + box_h - 1);
            const int clamped = std::max(min_y, std::min(max_y, y));
            const int rel = clamped - min_y;
            int mapped = scale_pose_coord_centered(rel, box_h, dst_h);
            return rotate_180 ? (int)dst_h - 1 - mapped : mapped;
        };

        int x1 = map_x(pose.x1);
        int y1 = map_y(pose.y1);
        int x2 = map_x(pose.x2);
        int y2 = map_y(pose.y2);
        pose.x1 = std::max(0, std::min((int)dst_w - 1, std::min(x1, x2)));
        pose.y1 = std::max(0, std::min((int)dst_h - 1, std::min(y1, y2)));
        pose.x2 = std::max(0, std::min((int)dst_w - 1, std::max(x1, x2)));
        pose.y2 = std::max(0, std::min((int)dst_h - 1, std::max(y1, y2)));

        for (auto &pt : pose.keypoints) {
            pt.x = std::max(0, std::min((int)dst_w - 1, map_x(pt.x)));
            pt.y = std::max(0, std::min((int)dst_h - 1, map_y(pt.y)));
        }
        out.push_back(pose);
    }
    return out;
}

static int detection_area(const dl::detect::result_t &det)
{
    if (det.box.size() < 4) {
        return 0;
    }
    return (det.box[2] - det.box[0]) * (det.box[3] - det.box[1]);
}

static bool get_largest_detection(const std::list<dl::detect::result_t> &dets, dl::detect::result_t *out)
{
    if (!out || dets.empty()) {
        return false;
    }

    auto best = std::max_element(dets.begin(),
                                 dets.end(),
                                 [](const dl::detect::result_t &a, const dl::detect::result_t &b) {
                                     return detection_area(a) < detection_area(b);
                                 });
    *out = *best;
    return true;
}

static bool clamp_pose_input_roi(pose_input_roi_t *roi, uint32_t frame_w, uint32_t frame_h)
{
    if (!roi || !roi->valid || frame_w == 0 || frame_h == 0 || roi->w == 0 || roi->h == 0) {
        return false;
    }

    uint32_t side = std::min(roi->w, roi->h);
    side = std::min(side, std::min(frame_w, frame_h));
    if (side < 32) {
        return false;
    }

    roi->w = side;
    roi->h = side;
    if (roi->x > frame_w - side) {
        roi->x = frame_w - side;
    }
    if (roi->y > frame_h - side) {
        roi->y = frame_h - side;
    }
    return true;
}

static float pose_track_pad_for_box(float box_w, float box_h, uint32_t frame_w, uint32_t frame_h)
{
    if (frame_w == 0 || frame_h == 0) {
        return PC_POSE_TRACK_CROP_PAD;
    }

    const float min_dim = (float)std::min(frame_w, frame_h);
    const float box_ratio = std::max(box_w, box_h) / std::max(1.0f, min_dim);
    if (box_ratio < 0.20f) {
        return 1.85f;
    }
    if (box_ratio < 0.35f) {
        return 2.10f;
    }
    if (box_ratio < 0.55f) {
        return 2.35f;
    }
    return PC_POSE_TRACK_CROP_PAD;
}

static bool make_pose_input_roi_from_bounds(int min_x,
                                            int min_y,
                                            int max_x,
                                            int max_y,
                                            uint32_t frame_w,
                                            uint32_t frame_h,
                                            pose_input_roi_t *out)
{
    if (!out || frame_w == 0 || frame_h == 0 || max_x <= min_x || max_y <= min_y) {
        return false;
    }

    min_x = std::max(0, std::min((int)frame_w - 1, min_x));
    min_y = std::max(0, std::min((int)frame_h - 1, min_y));
    max_x = std::max(0, std::min((int)frame_w - 1, max_x));
    max_y = std::max(0, std::min((int)frame_h - 1, max_y));
    if (max_x <= min_x || max_y <= min_y) {
        return false;
    }

    const float box_w = (float)(max_x - min_x + 1);
    const float box_h = (float)(max_y - min_y + 1);
    const float center_x = ((float)min_x + (float)max_x) * 0.5f;
    const float center_y = ((float)min_y + (float)max_y) * 0.5f;
    const float min_side = (float)std::min(frame_w, frame_h) * PC_POSE_TRACK_CROP_MIN_RATIO;
    const float max_side = (float)std::min(frame_w, frame_h);
    const float pad = pose_track_pad_for_box(box_w, box_h, frame_w, frame_h);
    float side_f = std::max(box_w, box_h) * pad;
    side_f = std::max(side_f, min_side);
    side_f = std::min(side_f, max_side);

    uint32_t side = (uint32_t)std::max(32, (int)lroundf(side_f));
    side = std::min(side, std::min(frame_w, frame_h));

    int x = (int)lroundf(center_x - (float)side * 0.5f);
    int y = (int)lroundf(center_y - (float)side * 0.5f);
    x = std::max(0, std::min((int)frame_w - (int)side, x));
    y = std::max(0, std::min((int)frame_h - (int)side, y));

    *out = {};
    out->valid = true;
    out->x = (uint32_t)x;
    out->y = (uint32_t)y;
    out->w = side;
    out->h = side;
    out->ts_us = esp_timer_get_time();
    return true;
}

static bool pose_result_to_input_roi(const pose_result_t &pose,
                                     uint32_t frame_w,
                                     uint32_t frame_h,
                                     pose_input_roi_t *out)
{
    if (!out || frame_w == 0 || frame_h == 0 || pose.keypoints.empty()) {
        return false;
    }

    int min_x = (int)frame_w - 1;
    int min_y = (int)frame_h - 1;
    int max_x = 0;
    int max_y = 0;
    int valid = 0;
    int upper = 0;
    const int n = std::min<int>(POSE_KPTS, (int)pose.keypoints.size());
    for (int k = 0; k < n; k++) {
        const pose_keypoint_t &kp = pose.keypoints[k];
        if (kp.score < PC_POSE_TRACK_KPT_SCORE_THR) {
            continue;
        }
        const int x = std::max(0, std::min((int)frame_w - 1, kp.x));
        const int y = std::max(0, std::min((int)frame_h - 1, kp.y));
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        valid++;
        if (k < POSE_KPT_LEG_START) {
            upper++;
        }
    }

    if (valid < PC_POSE_TRACK_MIN_KPTS || upper < PC_POSE_MIN_UPPER_KPTS) {
        return false;
    }

    if (pose.x2 > pose.x1 && pose.y2 > pose.y1) {
        min_x = std::min(min_x, std::max(0, std::min((int)frame_w - 1, pose.x1)));
        min_y = std::min(min_y, std::max(0, std::min((int)frame_h - 1, pose.y1)));
        max_x = std::max(max_x, std::max(0, std::min((int)frame_w - 1, pose.x2)));
        max_y = std::max(max_y, std::max(0, std::min((int)frame_h - 1, pose.y2)));
    }

    return make_pose_input_roi_from_bounds(min_x, min_y, max_x, max_y, frame_w, frame_h, out);
}

static bool pc_pose_roi_looks_misaligned(const pose_result_t &pose,
                                         uint32_t crop_x,
                                         uint32_t crop_y,
                                         uint32_t crop_w,
                                         uint32_t crop_h,
                                         bool *severe_out = nullptr)
{
    if (severe_out) {
        *severe_out = false;
    }
    if (crop_w < 32 || crop_h < 32 || pose.x2 <= pose.x1 || pose.y2 <= pose.y1) {
        return false;
    }

    const int rx1 = (int)crop_x;
    const int ry1 = (int)crop_y;
    const int rx2 = (int)crop_x + (int)crop_w - 1;
    const int ry2 = (int)crop_y + (int)crop_h - 1;
    const float crop_side = (float)std::max(crop_w, crop_h);
    const int edge_margin = std::max(6, (int)lroundf(crop_side * PC_POSE_ROI_EDGE_MARGIN_RATIO));
    const int severe_margin = std::max(3, (int)lroundf(crop_side * PC_POSE_ROI_SEVERE_EDGE_RATIO));

    const bool near_edge = pose.x1 <= rx1 + edge_margin || pose.y1 <= ry1 + edge_margin || pose.x2 >= rx2 - edge_margin || pose.y2 >= ry2 - edge_margin;
    const bool severe_edge = pose.x1 <= rx1 + severe_margin || pose.y1 <= ry1 + severe_margin || pose.x2 >= rx2 - severe_margin || pose.y2 >= ry2 - severe_margin;
    const float pose_cx = ((float)pose.x1 + (float)pose.x2) * 0.5f;
    const float pose_cy = ((float)pose.y1 + (float)pose.y2) * 0.5f;
    const float crop_cx = (float)crop_x + (float)crop_w * 0.5f;
    const float crop_cy = (float)crop_y + (float)crop_h * 0.5f;
    const float center_drift = (fabsf(pose_cx - crop_cx) + fabsf(pose_cy - crop_cy)) / std::max(1.0f, crop_side);
    const float box_ratio = (float)std::max(pose.x2 - pose.x1 + 1, pose.y2 - pose.y1 + 1) / std::max(1.0f, crop_side);

    if (severe_edge) {
        if (severe_out) {
            *severe_out = true;
        }
        return true;
    }
    if (!near_edge) {
        return false;
    }
    return center_drift > PC_POSE_ROI_DRIFT_RATIO ||
           (box_ratio < 0.60f && center_drift > PC_POSE_ROI_EDGE_DRIFT_RATIO);
}

static void clear_pose_tracking_roi(void)
{
    portENTER_CRITICAL(&s_pose_roi_lock);
    s_pose_track_roi = {};
    portEXIT_CRITICAL(&s_pose_roi_lock);
}

static void request_body_roi_reacquire_now(void)
{
    s_body_roi_reacquire_requested.store(true);
    s_last_body_roi_request_us.store(0);
}

static bool get_pose_tracking_roi(uint32_t frame_w, uint32_t frame_h, pose_input_roi_t *out)
{
    if (!PC_POSE_USE_ADAPTIVE_CROP || !out || frame_w == 0 || frame_h == 0) {
        return false;
    }

    pose_input_roi_t roi = {};
    portENTER_CRITICAL(&s_pose_roi_lock);
    roi = s_pose_track_roi;
    portEXIT_CRITICAL(&s_pose_roi_lock);

    const int64_t now_us = esp_timer_get_time();
    if (!roi.valid || now_us - roi.ts_us > PC_POSE_TRACK_CROP_MAX_AGE_US) {
        return false;
    }
    if (!clamp_pose_input_roi(&roi, frame_w, frame_h)) {
        return false;
    }
    *out = roi;
    return true;
}

static bool update_pose_tracking_roi_from_pose(const pose_result_t &pose,
                                               uint32_t frame_w,
                                               uint32_t frame_h,
                                               const char *source)
{
    pose_input_roi_t roi = {};
    if (!pose_result_to_input_roi(pose, frame_w, frame_h, &roi)) {
        return false;
    }

    const int64_t now_us = esp_timer_get_time();
    pose_input_roi_t prev = {};
    portENTER_CRITICAL(&s_pose_roi_lock);
    prev = s_pose_track_roi;
    if (prev.valid && now_us - prev.ts_us <= PC_POSE_TRACK_CROP_MAX_AGE_US) {
        const float prev_cx = (float)prev.x + (float)prev.w * 0.5f;
        const float prev_cy = (float)prev.y + (float)prev.h * 0.5f;
        const float roi_cx = (float)roi.x + (float)roi.w * 0.5f;
        const float roi_cy = (float)roi.y + (float)roi.h * 0.5f;
        const float move_ratio = (fabsf(roi_cx - prev_cx) + fabsf(roi_cy - prev_cy)) / std::max(1.0f, (float)prev.w);
        const float size_ratio = fabsf((float)roi.w - (float)prev.w) / std::max(1.0f, (float)prev.w);
        if (move_ratio <= 0.18f && size_ratio <= 0.18f) {
            const float alpha = 0.86f;
            roi.x = (uint32_t)std::max(0, (int)lroundf((1.0f - alpha) * (float)prev.x + alpha * (float)roi.x));
            roi.y = (uint32_t)std::max(0, (int)lroundf((1.0f - alpha) * (float)prev.y + alpha * (float)roi.y));
            uint32_t side = (uint32_t)std::max(32, (int)lroundf((1.0f - alpha) * (float)prev.w + alpha * (float)roi.w));
            side = std::min(side, std::min(frame_w, frame_h));
            roi.w = side;
            roi.h = side;
            clamp_pose_input_roi(&roi, frame_w, frame_h);
        }
    }
    roi.ts_us = now_us;
    s_pose_track_roi = roi;
    portEXIT_CRITICAL(&s_pose_roi_lock);

    static int64_t last_roi_log_us = 0;
    if (now_us - last_roi_log_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
        last_roi_log_us = now_us;
        ESP_LOGI(TAG,
                 "Pose body ROI update: %s roi=(%u,%u,%ux%u)",
                 source ? source : "pose",
                 (unsigned)roi.x,
                 (unsigned)roi.y,
                 (unsigned)roi.w,
                 (unsigned)roi.h);
    }
    return true;
}

static bool update_pose_tracking_roi_from_detection(const dl::detect::result_t &det,
                                                    uint32_t frame_w,
                                                    uint32_t frame_h,
                                                    const char *source)
{
    if (det.box.size() < 4 || frame_w == 0 || frame_h == 0 || det.score < BODY_ROI_DETECT_SCORE_THR) {
        return false;
    }

    const int box_w = det.box[2] - det.box[0];
    const int box_h = det.box[3] - det.box[1];
    const int min_box_px = std::max(16, (int)lroundf((float)std::min(frame_w, frame_h) * BODY_ROI_MIN_BOX_RATIO));
    if (box_w < min_box_px || box_h < min_box_px) {
        return false;
    }

    pose_input_roi_t roi = {};
    if (!make_pose_input_roi_from_bounds(det.box[0], det.box[1], det.box[2], det.box[3], frame_w, frame_h, &roi)) {
        return false;
    }

    const int64_t now_us = esp_timer_get_time();
    portENTER_CRITICAL(&s_pose_roi_lock);
    roi.ts_us = now_us;
    s_pose_track_roi = roi;
    portEXIT_CRITICAL(&s_pose_roi_lock);

    ESP_LOGI(TAG,
             "Pose body ROI reacquired: %s score=%.3f det=(%d,%d,%d,%d) roi=(%u,%u,%ux%u)",
             source ? source : "pedestrian",
             (double)det.score,
             det.box[0],
             det.box[1],
             det.box[2],
             det.box[3],
             (unsigned)roi.x,
             (unsigned)roi.y,
             (unsigned)roi.w,
             (unsigned)roi.h);
    return true;
}

static void set_face_overlay(bool visible, const dl::detect::result_t *det, const char *text)
{
    face_overlay_t overlay = {};
    overlay.visible = visible && det && det->box.size() >= 4;
    overlay.ts_us = esp_timer_get_time();
    if (overlay.visible) {
        overlay.x1 = det->box[0];
        overlay.y1 = det->box[1];
        overlay.x2 = det->box[2];
        overlay.y2 = det->box[3];
        snprintf(overlay.text, sizeof(overlay.text), "%s", text && text[0] ? text : "Face");
    }

    if (s_overlay_mutex && xSemaphoreTake(s_overlay_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        s_face_overlay = overlay;
        xSemaphoreGive(s_overlay_mutex);
    } else {
        s_face_overlay = overlay;
    }
}

static bool copy_face_overlay(face_overlay_t *out)
{
    if (!out) {
        return false;
    }

    if (s_overlay_mutex && xSemaphoreTake(s_overlay_mutex, pdMS_TO_TICKS(2)) == pdTRUE) {
        *out = s_face_overlay;
        xSemaphoreGive(s_overlay_mutex);
        return true;
    }

    *out = s_face_overlay;
    return true;
}

static void clear_pose_overlay(void)
{
    pose_overlay_t overlay = {};
    overlay.ts_us = esp_timer_get_time();
    s_pc_pose_overlay_hold = {};
    s_body_roi_reacquire_requested.store(false);
    s_last_pc_pose_success_us.store(0);
    reset_pc_pose_smooth();
    clear_pose_tracking_roi();
    if (s_pose_overlay_mutex && xSemaphoreTake(s_pose_overlay_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        s_pose_overlay_local = overlay;
        s_pose_overlay_pc = overlay;
        xSemaphoreGive(s_pose_overlay_mutex);
    } else {
        s_pose_overlay_local = overlay;
        s_pose_overlay_pc = overlay;
    }
}

static void clear_local_pose_overlay(void)
{
    pose_overlay_t overlay = {};
    overlay.ts_us = esp_timer_get_time();
    if (s_pose_overlay_mutex && xSemaphoreTake(s_pose_overlay_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        s_pose_overlay_local = overlay;
        xSemaphoreGive(s_pose_overlay_mutex);
    } else {
        s_pose_overlay_local = overlay;
    }
}

static pose_overlay_t make_pose_overlay(const std::vector<pose_result_t> &results,
                                        uint32_t width,
                                        uint32_t height,
                                        float kpt_visible_thr = POSE_OVERLAY_KPT_SCORE_THR,
                                        float kpt_box_thr = POSE_OVERLAY_KPT_SCORE_THR)
{
    pose_overlay_t overlay = {};
    overlay.ts_us = esp_timer_get_time();

    for (int i = 0; i < std::min<int>(POSE_MAX_PEOPLE, results.size()); i++) {
        const pose_result_t &src = results[i];
        pose_overlay_person_t &dst = overlay.people[overlay.count];
        dst.visible = true;
        dst.score = src.score;

        int min_x = (int)width - 1;
        int min_y = (int)height - 1;
        int max_x = 0;
        int max_y = 0;
        int visible_kpts = 0;
        const int keypoints = std::min<int>(POSE_KPTS, src.keypoints.size());
        for (int k = 0; k < keypoints; k++) {
            const float vis_thr = (k >= POSE_KPT_LEG_START) ? PC_POSE_LEG_KPT_SCORE_THR : kpt_visible_thr;
            dst.kpts[k].valid = src.keypoints[k].score >= vis_thr;
            dst.kpts[k].x = std::max(0, std::min((int)width - 1, src.keypoints[k].x));
            dst.kpts[k].y = std::max(0, std::min((int)height - 1, src.keypoints[k].y));
            if (k < POSE_KPT_LEG_START && src.keypoints[k].score >= kpt_box_thr) {
                min_x = std::min(min_x, dst.kpts[k].x);
                min_y = std::min(min_y, dst.kpts[k].y);
                max_x = std::max(max_x, dst.kpts[k].x);
                max_y = std::max(max_y, dst.kpts[k].y);
            }
            if (dst.kpts[k].valid) {
                visible_kpts++;
            }
        }

        if (visible_kpts < 1) {
            dst.visible = false;
            continue;
        }

        if (max_x <= min_x || max_y <= min_y) {
            min_x = (int)width - 1;
            min_y = (int)height - 1;
            max_x = 0;
            max_y = 0;
            for (int k = 0; k < keypoints; k++) {
                if (!dst.kpts[k].valid) {
                    continue;
                }
                min_x = std::min(min_x, dst.kpts[k].x);
                min_y = std::min(min_y, dst.kpts[k].y);
                max_x = std::max(max_x, dst.kpts[k].x);
                max_y = std::max(max_y, dst.kpts[k].y);
            }
        }

        const int pad = 20;
        if (max_x <= min_x || max_y <= min_y) {
            dst.x1 = std::max(0, min_x - pad);
            dst.y1 = std::max(0, min_y - pad);
            dst.x2 = std::min((int)width - 1, min_x + pad);
            dst.y2 = std::min((int)height - 1, min_y + pad);
        } else {
            dst.x1 = std::max(0, min_x - pad);
            dst.y1 = std::max(0, min_y - pad);
            dst.x2 = std::min((int)width - 1, max_x + pad);
            dst.y2 = std::min((int)height - 1, max_y + pad);
        }
        overlay.count++;
    }
    overlay.visible = overlay.count > 0;
    return overlay;
}

static void set_local_pose_overlay(const std::vector<pose_result_t> &results, uint32_t width, uint32_t height)
{
    pose_overlay_t overlay = make_pose_overlay(results, width, height, POSE_OVERLAY_KPT_SCORE_THR, PC_POSE_BOX_KPT_SCORE_THR);
    if (s_pose_overlay_mutex && xSemaphoreTake(s_pose_overlay_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        s_pose_overlay_local = overlay;
        xSemaphoreGive(s_pose_overlay_mutex);
    } else {
        s_pose_overlay_local = overlay;
    }
}

static bool local_pose_draw_quality_ok(const std::vector<pose_result_t> &results, uint32_t width, uint32_t height)
{
    if (results.empty() || width == 0 || height == 0 || results[0].keypoints.empty()) {
        return false;
    }

    const pose_result_t &pose = results[0];
    int visible = 0;
    int min_x = (int)width - 1;
    int min_y = (int)height - 1;
    int max_x = 0;
    int max_y = 0;
    const int n = std::min<int>(POSE_KPTS, (int)pose.keypoints.size());
    for (int k = 0; k < n; k++) {
        const pose_keypoint_t &kp = pose.keypoints[k];
        const float thr = (k >= POSE_KPT_LEG_START) ? PC_POSE_LEG_KPT_SCORE_THR : POSE_OVERLAY_KPT_SCORE_THR;
        if (kp.score < thr) {
            continue;
        }
        visible++;
        min_x = std::min(min_x, std::max(0, std::min((int)width - 1, kp.x)));
        min_y = std::min(min_y, std::max(0, std::min((int)height - 1, kp.y)));
        max_x = std::max(max_x, std::max(0, std::min((int)width - 1, kp.x)));
        max_y = std::max(max_y, std::max(0, std::min((int)height - 1, kp.y)));
    }

    const int min_box_px = std::max(24, (int)lroundf((float)std::min(width, height) * LOCAL_POSE_MIN_BOX_RATIO));
    return visible >= LOCAL_POSE_MIN_VISIBLE_KPTS &&
           max_x > min_x && max_y > min_y &&
           (max_x - min_x) >= min_box_px &&
           (max_y - min_y) >= min_box_px;
}

static void filter_pc_pose_suppress_lower_body(std::vector<pose_result_t> &results, uint32_t frame_h)
{
    if (results.empty() || frame_h < 32) {
        return;
    }

    pose_result_t &pose = results[0];
    if ((int)pose.keypoints.size() <= POSE_KPT_RIGHT_HIP) {
        return;
    }

    int shoulder_y = 0;
    int shoulder_n = 0;
    for (int idx : {POSE_KPT_LEFT_SHOULDER, POSE_KPT_RIGHT_SHOULDER}) {
        if (pose.keypoints[idx].score >= PC_POSE_PARSE_SCORE_THR) {
            shoulder_y = std::max(shoulder_y, pose.keypoints[idx].y);
            shoulder_n++;
        }
    }
    if (shoulder_n == 0) {
        return;
    }

    const int leg_margin = std::max(28, (int)frame_h / 10);
    const int leg_y_limit = std::min((int)frame_h - 1, shoulder_y + leg_margin);

    for (int k = POSE_KPT_LEG_START; k < (int)pose.keypoints.size(); k++) {
        pose_keypoint_t &kp = pose.keypoints[k];
        if (kp.score <= 0.0f) {
            continue;
        }
        if (kp.y > leg_y_limit || kp.score < PC_POSE_LEG_KPT_SCORE_THR) {
            kp.score = 0.0f;
        }
    }
}

static int count_pc_pose_upper_visible(const pose_result_t &pose)
{
    int count = 0;
    const int n = std::min<int>(POSE_KPT_LEG_START, (int)pose.keypoints.size());
    for (int k = 0; k < n; k++) {
        if (pose.keypoints[k].score >= POSE_OVERLAY_KPT_SCORE_THR) {
            count++;
        }
    }
    return count;
}

static int pc_pose_lerp_i(int prev, int cur, float alpha)
{
    return (int)lroundf((1.0f - alpha) * (float)prev + alpha * (float)cur);
}

static void reset_pc_pose_smooth(void)
{
    s_pc_pose_smooth_ready = false;
    s_pc_pose_smooth_prev = {};
}

static void smooth_pc_pose_results(std::vector<pose_result_t> &results, uint32_t frame_w, uint32_t frame_h)
{
    if (results.empty() || frame_w == 0 || frame_h == 0) {
        return;
    }

    pose_result_t &cur = results[0];
    if (!s_pc_pose_smooth_ready) {
        s_pc_pose_smooth_prev = cur;
        s_pc_pose_smooth_ready = true;
        return;
    }

    pose_result_t &prev = s_pc_pose_smooth_prev;

    cur.x1 = pc_pose_lerp_i(prev.x1, cur.x1, PC_POSE_SMOOTH_ALPHA_BOX);
    cur.y1 = pc_pose_lerp_i(prev.y1, cur.y1, PC_POSE_SMOOTH_ALPHA_BOX);
    cur.x2 = pc_pose_lerp_i(prev.x2, cur.x2, PC_POSE_SMOOTH_ALPHA_BOX);
    cur.y2 = pc_pose_lerp_i(prev.y2, cur.y2, PC_POSE_SMOOTH_ALPHA_BOX);
    cur.score = 0.85f * prev.score + 0.15f * cur.score;

    const size_t nk = std::min(cur.keypoints.size(), prev.keypoints.size());
    for (size_t k = 0; k < nk; k++) {
        pose_keypoint_t &ck = cur.keypoints[k];
        const pose_keypoint_t &pk = prev.keypoints[k];
        if (k >= (size_t)POSE_KPT_LEG_START) {
            if (ck.score < PC_POSE_LEG_KPT_SCORE_THR) {
                ck.score = 0.0f;
            } else if (pk.score >= PC_POSE_LEG_KPT_SCORE_THR) {
                ck.x = pc_pose_lerp_i(pk.x, ck.x, PC_POSE_SMOOTH_ALPHA_KPT);
                ck.y = pc_pose_lerp_i(pk.y, ck.y, PC_POSE_SMOOTH_ALPHA_KPT);
            }
            continue;
        }
        if (ck.score >= PC_POSE_PARSE_SCORE_THR || pk.score >= POSE_OVERLAY_KPT_SCORE_THR) {
            ck.x = pc_pose_lerp_i(pk.x, ck.x, PC_POSE_SMOOTH_ALPHA_KPT);
            ck.y = pc_pose_lerp_i(pk.y, ck.y, PC_POSE_SMOOTH_ALPHA_KPT);
            ck.score = std::max(ck.score, pk.score * 0.95f);
        }
    }

    prev = cur;
}

static void reset_offline_draw_smooth(void)
{
    s_offline_draw_smooth_ready = false;
    s_offline_draw_smooth_prev = {};
}

static void reset_offline_analysis_smooth(void)
{
    s_offline_analysis_smooth_ready = false;
    s_offline_analysis_smooth_prev = {};
    memset(s_offline_analysis_kpt_miss, 0, sizeof(s_offline_analysis_kpt_miss));
}

static bool offline_action_needs_lower_body(const std::string &action)
{
    return action == "squat" || action == "deadlift" || action == "situp" || action == "plank";
}

static float offline_analysis_min_score_for_kpt(const std::string &action, size_t k)
{
    if (k >= (size_t)POSE_KPT_LEFT_HIP && k <= (size_t)POSE_KPT_RIGHT_ANKLE) {
        return offline_action_needs_lower_body(action) ? 0.045f : 0.060f;
    }
    if (k >= (size_t)POSE_KPT_LEFT_SHOULDER && k <= (size_t)POSE_KPT_RIGHT_WRIST) {
        return 0.060f;
    }
    return 0.070f;
}

static pose_result_t smooth_offline_pose_for_analysis(const pose_result_t &src, const std::string &action)
{
    pose_result_t out = src;
    if (!s_offline_analysis_smooth_ready ||
        s_offline_analysis_smooth_prev.keypoints.size() != src.keypoints.size()) {
        s_offline_analysis_smooth_prev = src;
        s_offline_analysis_smooth_ready = true;
        memset(s_offline_analysis_kpt_miss, 0, sizeof(s_offline_analysis_kpt_miss));
        return out;
    }

    pose_result_t &prev = s_offline_analysis_smooth_prev;
    const int box_w = std::max(1, src.x2 - src.x1);
    const int box_h = std::max(1, src.y2 - src.y1);
    const float diag = sqrtf((float)box_w * (float)box_w + (float)box_h * (float)box_h);
    const float max_jump = std::max(90.0f, diag * 0.34f);

    out.x1 = pc_pose_lerp_i(prev.x1, src.x1, 0.54f);
    out.y1 = pc_pose_lerp_i(prev.y1, src.y1, 0.54f);
    out.x2 = pc_pose_lerp_i(prev.x2, src.x2, 0.54f);
    out.y2 = pc_pose_lerp_i(prev.y2, src.y2, 0.54f);
    out.score = std::max(src.score, prev.score * 0.82f);

    const size_t nk = std::min(src.keypoints.size(), (size_t)POSE_KPTS);
    for (size_t k = 0; k < nk; ++k) {
        const pose_keypoint_t &ck = src.keypoints[k];
        const pose_keypoint_t &pk = prev.keypoints[k];
        const float min_score = offline_analysis_min_score_for_kpt(action, k);
        const bool has_cur = ck.score >= min_score;
        const bool has_prev = pk.score >= min_score;
        const bool is_lower_kpt = k >= (size_t)POSE_KPT_LEFT_HIP && k <= (size_t)POSE_KPT_RIGHT_ANKLE;
        const uint8_t max_hold = (offline_action_needs_lower_body(action) && is_lower_kpt) ? 3 : 2;

        if (!has_cur) {
            if (has_prev && s_offline_analysis_kpt_miss[k] < max_hold) {
                out.keypoints[k] = pk;
                out.keypoints[k].score = std::max(min_score, pk.score * 0.80f);
                s_offline_analysis_kpt_miss[k]++;
            } else {
                out.keypoints[k].score = 0.0f;
                s_offline_analysis_kpt_miss[k] = 3;
            }
            continue;
        }

        s_offline_analysis_kpt_miss[k] = 0;
        if (!has_prev) {
            out.keypoints[k] = ck;
            continue;
        }

        const float dx = (float)(ck.x - pk.x);
        const float dy = (float)(ck.y - pk.y);
        const float dist = sqrtf(dx * dx + dy * dy);
        if (dist > max_jump && ck.score < 0.22f) {
            out.keypoints[k] = pk;
            out.keypoints[k].score = std::max(min_score, pk.score * 0.78f);
            continue;
        }

        const float score_alpha = std::max(0.0f, std::min(1.0f, (ck.score - min_score) / 0.30f));
        const float jump_alpha = std::max(0.0f, std::min(1.0f, dist / max_jump));
        const float alpha = std::max(0.32f, std::min(0.68f, 0.32f + score_alpha * 0.22f + jump_alpha * 0.14f));
        out.keypoints[k].x = pc_pose_lerp_i(pk.x, ck.x, alpha);
        out.keypoints[k].y = pc_pose_lerp_i(pk.y, ck.y, alpha);
        out.keypoints[k].score = std::max(ck.score, pk.score * 0.72f);
    }

    prev = out;
    return out;
}

static pose_result_t smooth_offline_pose_for_draw(const pose_result_t &src)
{
    s_offline_draw_smooth_prev = src;
    s_offline_draw_smooth_ready = true;
    return src;
}

static void set_pc_pose_overlay(const std::vector<pose_result_t> &results, uint32_t width, uint32_t height)
{
    pose_overlay_t overlay = make_pose_overlay(results, width, height, POSE_OVERLAY_KPT_SCORE_THR, PC_POSE_BOX_KPT_SCORE_THR);
    if (!overlay.visible) {
        return;
    }
    if (s_pose_overlay_mutex && xSemaphoreTake(s_pose_overlay_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        s_pose_overlay_pc = overlay;
        s_pc_pose_overlay_hold = overlay;
        xSemaphoreGive(s_pose_overlay_mutex);
    } else {
        s_pose_overlay_pc = overlay;
        s_pc_pose_overlay_hold = overlay;
    }
}

static void clear_pc_pose_overlay(void)
{
    pose_overlay_t overlay = {};
    overlay.ts_us = esp_timer_get_time();
    if (s_pose_overlay_mutex && xSemaphoreTake(s_pose_overlay_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        s_pose_overlay_pc = overlay;
        s_pc_pose_overlay_hold = overlay;
        xSemaphoreGive(s_pose_overlay_mutex);
    } else {
        s_pose_overlay_pc = overlay;
        s_pc_pose_overlay_hold = overlay;
    }
}

static void refresh_pc_pose_overlay_hold(void)
{
    if (!s_pc_pose_overlay_hold.visible) {
        return;
    }
    s_pc_pose_overlay_hold.ts_us = esp_timer_get_time();
    if (s_pose_overlay_mutex && xSemaphoreTake(s_pose_overlay_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        s_pose_overlay_pc = s_pc_pose_overlay_hold;
        xSemaphoreGive(s_pose_overlay_mutex);
    } else {
        s_pose_overlay_pc = s_pc_pose_overlay_hold;
    }
}

static void note_pc_pose_miss(int *miss_count)
{
    if (!miss_count) {
        return;
    }
    (*miss_count)++;
    if (*miss_count >= PC_POSE_MISSES_BEFORE_ROI_REACQUIRE) {
        request_body_roi_reacquire_now();
    }
    if (*miss_count >= PC_POSE_MISSES_BEFORE_OVERLAY_CLEAR) {
        reset_pc_pose_smooth();
        clear_pc_pose_overlay();
        *miss_count = 0;
    }
}

static bool copy_pose_overlays(pose_overlay_t *local, pose_overlay_t *pc)
{
    if (!local || !pc) {
        return false;
    }

    if (s_pose_overlay_mutex && xSemaphoreTake(s_pose_overlay_mutex, pdMS_TO_TICKS(2)) == pdTRUE) {
        *local = s_pose_overlay_local;
        *pc = s_pose_overlay_pc;
        xSemaphoreGive(s_pose_overlay_mutex);
        return true;
    }

    *local = s_pose_overlay_local;
    *pc = s_pose_overlay_pc;
    return true;
}

static void draw_pixel_rgb565(uint16_t *px, uint32_t stride, uint32_t w, uint32_t h, int x, int y, uint16_t color)
{
    if (x >= 0 && y >= 0 && x < (int)w && y < (int)h) {
        px[y * stride + x] = color;
    }
}

static void draw_point_rgb565(uint16_t *px, uint32_t stride, uint32_t w, uint32_t h, int x, int y, uint16_t color)
{
    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            if (dx * dx + dy * dy <= 9) {
                draw_pixel_rgb565(px, stride, w, h, x + dx, y + dy, color);
            }
        }
    }
}

static void draw_line_rgb565(uint16_t *px,
                             uint32_t stride,
                             uint32_t w,
                             uint32_t h,
                             int x0,
                             int y0,
                             int x1,
                             int y1,
                             uint16_t color)
{
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;

    for (;;) {
        draw_point_rgb565(px, stride, w, h, x0, y0, color);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

static void draw_rect_rgb565(uint16_t *px,
                             uint32_t stride,
                             uint32_t w,
                             uint32_t h,
                             int x1,
                             int y1,
                             int x2,
                             int y2,
                             uint16_t color)
{
    x1 = std::max(0, std::min((int)w - 1, x1));
    y1 = std::max(0, std::min((int)h - 1, y1));
    x2 = std::max(0, std::min((int)w - 1, x2));
    y2 = std::max(0, std::min((int)h - 1, y2));
    if (x2 <= x1 || y2 <= y1) {
        return;
    }

    for (int t = 0; t < 3; t++) {
        for (int x = x1; x <= x2; x++) {
            draw_pixel_rgb565(px, stride, w, h, x, y1 + t, color);
            draw_pixel_rgb565(px, stride, w, h, x, y2 - t, color);
        }
        for (int y = y1; y <= y2; y++) {
            draw_pixel_rgb565(px, stride, w, h, x1 + t, y, color);
            draw_pixel_rgb565(px, stride, w, h, x2 - t, y, color);
        }
    }
}

static void draw_face_overlay(uint8_t *buf, uint32_t w, uint32_t h, size_t len)
{
    face_overlay_t overlay = {};
    if (!buf || !copy_face_overlay(&overlay) || !overlay.visible ||
        esp_timer_get_time() - overlay.ts_us > FACE_OVERLAY_TIMEOUT_US) {
        return;
    }

    int x1 = std::max(0, std::min((int)w - 1, overlay.x1));
    int y1 = std::max(0, std::min((int)h - 1, overlay.y1));
    int x2 = std::max(0, std::min((int)w - 1, overlay.x2));
    int y2 = std::max(0, std::min((int)h - 1, overlay.y2));
    if (x2 <= x1 || y2 <= y1) {
        return;
    }

    const uint32_t stride = get_frame_stride_pixels(w, h, len);
    uint16_t *px = reinterpret_cast<uint16_t *>(buf);
    const uint16_t green = 0x07e0;
    const int thickness = 4;

    for (int t = 0; t < thickness; t++) {
        const int yt = y1 + t;
        const int yb = y2 - t;
        if (yt >= 0 && yt < (int)h) {
            uint16_t *row = px + yt * stride;
            for (int x = x1; x <= x2; x++) {
                row[x] = green;
            }
        }
        if (yb >= 0 && yb < (int)h) {
            uint16_t *row = px + yb * stride;
            for (int x = x1; x <= x2; x++) {
                row[x] = green;
            }
        }
        const int xl = x1 + t;
        const int xr = x2 - t;
        if (xl >= 0 && xl < (int)w) {
            for (int y = y1; y <= y2; y++) {
                px[y * stride + xl] = green;
            }
        }
        if (xr >= 0 && xr < (int)w) {
            for (int y = y1; y <= y2; y++) {
                px[y * stride + xr] = green;
            }
        }
    }
}

static void draw_pose_overlay_layer(uint16_t *px,
                                    uint32_t stride,
                                    uint32_t w,
                                    uint32_t h,
                                    const pose_overlay_t &overlay,
                                    uint16_t box_color,
                                    uint16_t bone_color,
                                    uint16_t point_color)
{
    static const int skeleton[][2] = {
        {0, 1},   {1, 2},   {2, 3},   {3, 7},   {0, 4},   {4, 5},
        {5, 6},   {6, 8},   {9, 10},  {11, 12}, {11, 13}, {13, 15},
        {15, 17}, {15, 19}, {15, 21}, {17, 19}, {12, 14}, {14, 16},
        {16, 18}, {16, 20}, {16, 22}, {18, 20}, {11, 23}, {12, 24},
        {23, 24}, {23, 25}, {25, 27}, {27, 29}, {27, 31}, {29, 31},
        {24, 26}, {26, 28}, {28, 30}, {28, 32}, {30, 32},
    };

    for (int i = 0; i < overlay.count; i++) {
        const pose_overlay_person_t &person = overlay.people[i];
        if (!person.visible) {
            continue;
        }

        draw_rect_rgb565(px, stride, w, h, person.x1, person.y1, person.x2, person.y2, box_color);
        for (const auto &bone : skeleton) {
            const pose_overlay_point_t &a = person.kpts[bone[0]];
            const pose_overlay_point_t &b = person.kpts[bone[1]];
            if (a.valid && b.valid) {
                draw_line_rgb565(px, stride, w, h, a.x, a.y, b.x, b.y, bone_color);
            }
        }
        for (int k = 0; k < POSE_KPTS; k++) {
            const pose_overlay_point_t &pt = person.kpts[k];
            if (pt.valid) {
                draw_point_rgb565(px, stride, w, h, pt.x, pt.y, point_color);
            }
        }
    }
}

static void draw_pose_overlay(uint8_t *buf, uint32_t w, uint32_t h, size_t len)
{
    pose_overlay_t local = {};
    pose_overlay_t pc = {};
    pose_overlay_t hold = {};
    if (!buf) {
        return;
    }
    if (s_pose_overlay_mutex && xSemaphoreTake(s_pose_overlay_mutex, pdMS_TO_TICKS(2)) == pdTRUE) {
        local = s_pose_overlay_local;
        pc = s_pose_overlay_pc;
        hold = s_pc_pose_overlay_hold;
        xSemaphoreGive(s_pose_overlay_mutex);
    } else {
        local = s_pose_overlay_local;
        pc = s_pose_overlay_pc;
        hold = s_pc_pose_overlay_hold;
    }

    const int64_t now = esp_timer_get_time();
    const uint32_t stride = get_frame_stride_pixels(w, h, len);
    uint16_t *px = reinterpret_cast<uint16_t *>(buf);

    ESP_LOGD(TAG, "pose draw state: pc=%d local=%d age_pc=%lld age_local=%lld",
             pc.visible ? 1 : 0,
             local.visible ? 1 : 0,
             (long long)(now - pc.ts_us),
             (long long)(now - local.ts_us));
    pose_overlay_t draw_pc = pc;
    if (hold.visible && (!draw_pc.visible || hold.ts_us >= draw_pc.ts_us)) {
        draw_pc = hold;
    }
    if (draw_pc.visible && now - draw_pc.ts_us <= PC_POSE_OVERLAY_HOLD_US) {
        draw_pose_overlay_layer(px, stride, w, h, draw_pc, 0x07e0, 0x07e0, 0x07e0);
    }
    if (local.visible && now - local.ts_us <= FACE_OVERLAY_TIMEOUT_US) {
        draw_pose_overlay_layer(px, stride, w, h, local, 0xf800, 0xf800, 0xf800);
    }
    static int pose_draw_log_count = 0;
    if (pose_draw_log_count < 8) {
        pose_draw_log_count++;
        ESP_LOGI(TAG, "Pose draw state: pc=%d local=%d age_pc=%lld age_local=%lld",
                 pc.visible ? 1 : 0,
                 local.visible ? 1 : 0,
                 (long long)(now - pc.ts_us),
                 (long long)(now - local.ts_us));
    }

    static int64_t last_empty_overlay_log_us = 0;
    if (!draw_pc.visible && !local.visible &&
        now - last_empty_overlay_log_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
        last_empty_overlay_log_us = now;
        ESP_LOGW(TAG, "Pose overlay empty: no visible results to draw");
    }
}

static void update_db_count_label(void)
{
    if (!lbl_db_count) {
        return;
    }

    char line[32];
    snprintf(line, sizeof(line), "特征库: %d", s_db_count.load());
    lv_label_set_text(lbl_db_count, line);
}

static void update_wifi_labels_ui(void)
{
    static int64_t last_netif_probe_us = 0;
    const int64_t now_us = esp_timer_get_time();
    const WifiUiState current_wifi_state = (WifiUiState)s_wifi_ui_state.load();
    if (!s_pc_pose_wifi_ready.load() && s_pc_pose_wifi_started.load() &&
        current_wifi_state == WifiUiState::Connecting &&
        now_us - last_netif_probe_us >= 1000000) {
        last_netif_probe_us = now_us;
        wifi_poll_netif_ip("ui");
    }
    if (s_pc_pose_wifi_ready.load() && s_wifi_ip[0] != '\0' &&
        (WifiUiState)s_wifi_ui_state.load() != WifiUiState::Connected) {
        s_wifi_ui_state.store((int)WifiUiState::Connected);
    }

    if (lbl_wifi_state) {
        char line[64];
        snprintf(line, sizeof(line), "状态: %s", wifi_ui_state_text());
        lv_label_set_text(lbl_wifi_state, line);
        lv_obj_set_style_text_color(lbl_wifi_state, lv_color_hex(wifi_ui_state_color()), LV_PART_MAIN);
    }
    if (lbl_wifi_ip) {
        char line[64];
        snprintf(line, sizeof(line), "IP: %s", s_wifi_ip[0] ? s_wifi_ip : "--");
        lv_label_set_text(lbl_wifi_ip, line);
    }
    if (lbl_wifi_saved_ssid) {
        char line[72];
        snprintf(line, sizeof(line), "当前 SSID: %s", s_wifi_ssid[0] ? s_wifi_ssid : "--");
        lv_label_set_text(lbl_wifi_saved_ssid, line);
    }
    if (lbl_menu_wifi) {
        char line[80];
        snprintf(line, sizeof(line), "Wi-Fi: %s / %s",
                 wifi_ui_state_text(),
                 s_wifi_ssid[0] ? s_wifi_ssid : "--");
        lv_label_set_text(lbl_menu_wifi, line);
    }
    if (lbl_menu_pc) {
        const int64_t last_pc_ok = s_last_pc_pose_success_us.load();
        const bool pc_recent = last_pc_ok > 0 && now_us - last_pc_ok < 10000000LL;
        lv_label_set_text(lbl_menu_pc,
                          !s_pc_pose_wifi_ready.load() ? "PC服务: 等待网络" :
                          (pc_recent ? "PC服务: 最近已响应" : "PC服务: 网络就绪，等待请求"));
    }
    if (lbl_menu_ip) {
        char line[80];
        snprintf(line, sizeof(line), "IP: %s", s_wifi_ip[0] ? s_wifi_ip : "--");
        lv_label_set_text(lbl_menu_ip, line);
    }
}

static void update_settings_status_ui(void)
{
    if (lbl_settings_account) {
        char line[96];
        snprintf(line,
                 sizeof(line),
                 "账号: %s%s",
                 s_auth_has_password.load() ? s_auth_saved_user : "未注册",
                 s_auth_has_password.load() ? " / 密码已保存" : "");
        lv_label_set_text(lbl_settings_account, line);
    }
    if (lbl_settings_face) {
        char line[80];
        if (s_rec) {
            snprintf(line, sizeof(line), "人脸库: %d 个特征", s_db_count.load());
        } else {
            snprintf(line, sizeof(line), "人脸库: 未加载");
        }
        lv_label_set_text(lbl_settings_face, line);
    }
    if (lbl_settings_profile) {
        char line[120];
        user_profile_summary(line, sizeof(line));
        lv_label_set_text(lbl_settings_profile, line);
    }
    if (lbl_settings_heap) {
        char line[96];
        const size_t internal_kb = heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024;
        const size_t psram_kb = heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024;
        snprintf(line, sizeof(line), "内存: IRAM %uKB / PSRAM %uKB", (unsigned)internal_kb, (unsigned)psram_kb);
        lv_label_set_text(lbl_settings_heap, line);
    }
    if (lbl_settings_uptime) {
        char line[80];
        const uint32_t sec = (uint32_t)(esp_timer_get_time() / 1000000ULL);
        snprintf(line, sizeof(line), "运行: %02u:%02u:%02u", sec / 3600, (sec / 60) % 60, sec % 60);
        lv_label_set_text(lbl_settings_uptime, line);
    }
    if (lbl_settings_pc) {
        char line[96];
        const int64_t now_us = esp_timer_get_time();
        const int64_t last_pc_ok = s_last_pc_pose_success_us.load();
        const bool pc_recent = last_pc_ok > 0 && now_us - last_pc_ok < 10000000LL;
        snprintf(line,
                 sizeof(line),
                 "PC服务: %s  %s",
                 !s_pc_pose_wifi_ready.load() ? "等待网络" : (pc_recent ? "最近已响应" : "等待请求"),
                 s_wifi_ip[0] ? s_wifi_ip : "--");
        lv_label_set_text(lbl_settings_pc, line);
    }
}

static void auth_enter_main_ui(void)
{
    s_auth_face_login_active.store(false);
    s_auth_unlock_pending.store(false);
    s_live_rec = false;
    s_live_busy.store(false);
    s_last_recognition_us.store(0);
    if (kb_auth) {
        lv_obj_add_flag(kb_auth, LV_OBJ_FLAG_HIDDEN);
        lv_keyboard_set_textarea(kb_auth, nullptr);
    }

    if (lbl_auth_status) {
        char line[80];
        snprintf(line, sizeof(line), "欢迎 %s", s_auth_user[0] ? s_auth_user : "用户");
        lv_label_set_text(lbl_auth_status, line);
    }

    if (!s_user_profile.complete) {
        profile_sync_edit_state();
        g_ui_screen.store(static_cast<int>(UiScreen::Profile));
        if (!scr_profile) {
            create_page_checked(&scr_profile, create_profile_page, "profile");
        }
        if (scr_profile) {
            lv_scr_load_anim(scr_profile, LV_SCREEN_LOAD_ANIM_FADE_ON, 260, 0, false);
            ESP_LOGI(TAG, "Auth complete, enter profile setup as %s", s_auth_user[0] ? s_auth_user : "user");
            return;
        }
    }

    g_ui_screen.store(static_cast<int>(UiScreen::Menu));
    if (!scr_menu) {
        create_page_checked(&scr_menu, create_menu, "overview");
    }
    if (scr_menu) {
        lv_scr_load_anim(scr_menu, LV_SCREEN_LOAD_ANIM_FADE_ON, 260, 0, false);
        ESP_LOGI(TAG, "Auth complete, enter main UI as %s", s_auth_user[0] ? s_auth_user : "user");
    } else {
        g_ui_screen.store(static_cast<int>(UiScreen::Auth));
        if (lbl_auth_status) {
            lv_label_set_text(lbl_auth_status, "主界面准备失败，正在重试");
        }
    }
}

static void update_face_overlay_ui(void)
{
    if (!box_face || !lbl_face_name) {
        return;
    }

    face_overlay_t overlay = {};
    const bool visible = copy_face_overlay(&overlay) && overlay.visible &&
                         g_ui_screen.load() == static_cast<int>(UiScreen::Face) &&
                         esp_timer_get_time() - overlay.ts_us <= FACE_OVERLAY_TIMEOUT_US;
    if (!visible || s_cam_w == 0 || s_cam_h == 0) {
        lv_obj_add_flag(box_face, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(lbl_face_name, LV_OBJ_FLAG_HIDDEN);
        return;
    }

    const int32_t canvas_x = 0;
    const int32_t canvas_y = 0;
    int32_t x1 = canvas_x + scale_coord(overlay.x1, s_cam_w, PREVIEW_W);
    int32_t y1 = canvas_y + scale_coord(overlay.y1, s_cam_h, PREVIEW_H);
    int32_t x2 = canvas_x + scale_coord(overlay.x2, s_cam_w, PREVIEW_W);
    int32_t y2 = canvas_y + scale_coord(overlay.y2, s_cam_h, PREVIEW_H);

    x1 = std::max<int32_t>(0, std::min<int32_t>(PREVIEW_W - 1, x1));
    y1 = std::max<int32_t>(0, std::min<int32_t>(PREVIEW_H - 1, y1));
    x2 = std::max<int32_t>(0, std::min<int32_t>(PREVIEW_W - 1, x2));
    y2 = std::max<int32_t>(0, std::min<int32_t>(PREVIEW_H - 1, y2));
    if (x2 <= x1 + 4 || y2 <= y1 + 4) {
        lv_obj_add_flag(box_face, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(lbl_face_name, LV_OBJ_FLAG_HIDDEN);
        return;
    }

    lv_obj_clear_flag(box_face, LV_OBJ_FLAG_HIDDEN);
    lv_obj_set_pos(box_face, x1, y1);
    lv_obj_set_size(box_face, x2 - x1, y2 - y1);

    lv_label_set_text(lbl_face_name, overlay.text);
    lv_obj_clear_flag(lbl_face_name, LV_OBJ_FLAG_HIDDEN);
    lv_obj_set_pos(lbl_face_name, x1, std::max<int32_t>(0, y1 - 32));
}

static void update_preview_canvas_ui(void)
{
    if (!s_preview_dirty.exchange(false) || !s_preview_buf) {
        return;
    }

    lv_obj_t *canvas = nullptr;
    lv_obj_t *view = nullptr;
    const int screen = g_ui_screen.load();
    if (screen == static_cast<int>(UiScreen::Camera)) {
        canvas = cv_cam;
        view = view_cam;
    } else if (screen == static_cast<int>(UiScreen::Auth)) {
        canvas = cv_auth;
        view = view_auth;
    } else if (screen == static_cast<int>(UiScreen::Face)) {
        canvas = cv_face;
        view = view_face;
    } else if (screen == static_cast<int>(UiScreen::Pose)) {
        canvas = cv_pose;
        view = view_pose;
    }

    if (!canvas || !view) {
        return;
    }

    lv_canvas_set_buffer(canvas, s_preview_buf, PREVIEW_W, PREVIEW_H, LV_COLOR_FORMAT_RGB565);
    place_canvas(canvas, view, PREVIEW_W, PREVIEW_H);
    lv_obj_invalidate(canvas);
}

static void format_mmss(uint32_t ms, char *out, size_t out_len)
{
    const uint32_t sec = ms / 1000;
    snprintf(out, out_len, "%02u:%02u", (unsigned)(sec / 60), (unsigned)(sec % 60));
}

static void training_lock(void)
{
    if (s_training_mutex) {
        xSemaphoreTake(s_training_mutex, portMAX_DELAY);
    }
}

static void training_unlock(void)
{
    if (s_training_mutex) {
        xSemaphoreGive(s_training_mutex);
    }
}

static const training_profile_t *training_profile_by_index(int index)
{
    if (index < 0 || index >= TRAINING_PROFILE_COUNT) {
        index = 0;
    }
    return &TRAINING_PROFILES[index];
}

static int training_profile_index(const training_profile_t *profile)
{
    if (!profile) {
        return 0;
    }
    for (int i = 0; i < TRAINING_PROFILE_COUNT; ++i) {
        if (profile == &TRAINING_PROFILES[i]) {
            return i;
        }
    }
    return 0;
}

static int training_action_index_from_id(const std::string &action)
{
    if (action == "squat") {
        return 0;
    }
    if (action == "pushup") {
        return 1;
    }
    if (action == "bench_press") {
        return 2;
    }
    if (action == "pullup") {
        return 3;
    }
    if (action == "deadlift") {
        return 4;
    }
    if (action == "plank") {
        return 5;
    }
    if (action == "dumbbell_curl" || action == "curl") {
        return 6;
    }
    if (action == "situp") {
        return 7;
    }
    return 0;
}

static const char *training_action_id_from_index(int index)
{
    switch (index) {
    case 1:
        return "pushup";
    case 2:
        return "bench_press";
    case 3:
        return "pullup";
    case 4:
        return "deadlift";
    case 5:
        return "plank";
    case 6:
        return "dumbbell_curl";
    case 7:
        return "situp";
    case 0:
    default:
        return "squat";
    }
}

static const char *training_action_guide_for_index(int index)
{
    switch (index) {
    case 1:
        return "指导: 侧前方45°拍全身；胸肩髋踝尽量一条线；下降到肘角明显变小再推起，塌腰和耸肩会扣分。";
    case 2:
        return "指导: 斜侧方拍上半身和杠铃路径；手腕肘肩同步，下降到胸前再推起；避免肘外翻和半程。";
    case 3:
        return "指导: 正前或斜前方拍到头、肩、髋；下放伸展再上拉到下巴接近杠；摆动过大或半程会扣分。";
    case 4:
        return "指导: 侧方拍全身和脚下；髋膝一起伸展，背部保持稳定；弓背、耸肩和膝髋不同步会扣分。";
    case 5:
        return "指导: 侧方拍肩髋踝；进入支撑后开始计时；塌腰或离开支撑会暂停，抬臀过高会提示纠正。";
    case 6:
        return "指导: 正前或斜前方拍上半身；上臂尽量固定，完整屈肘再控制下放；借力摆动会扣分。";
    case 7:
        return "指导: 侧方拍躯干和髋部；腹部卷起再控制下放；猛拉颈部、幅度过小和速度过快会扣分。";
    case 0:
    default:
        return "指导: 侧前方45°拍全身；髋膝踝都要入镜；下蹲到髋部明显下降再站起，膝盖跟脚尖同向。";
    }
}

static void training_goal_set_defaults(void)
{
    for (int i = 0; i < TRAINING_PROFILE_COUNT; ++i) {
        const training_profile_t *profile = training_profile_by_index(i);
        s_training_goal_sets[i] = std::max<uint8_t>(1, profile->sets);
        s_training_goal_reps[i] = std::max<uint8_t>(1, profile->reps_per_set);
        s_training_goal_duration_sec[i] = std::max<uint16_t>(15, profile->duration_sec);
    }
}

static int training_goal_sets_for_index(int index)
{
    if (index < 0 || index >= TRAINING_PROFILE_COUNT) {
        index = 0;
    }
    return std::max(1, (int)s_training_goal_sets[index]);
}

static int training_goal_reps_for_index(int index)
{
    if (index < 0 || index >= TRAINING_PROFILE_COUNT) {
        index = 0;
    }
    return std::max(1, (int)s_training_goal_reps[index]);
}

static uint32_t training_goal_duration_for_index(int index)
{
    if (index < 0 || index >= TRAINING_PROFILE_COUNT) {
        index = 0;
    }
    return std::max<uint32_t>(15, s_training_goal_duration_sec[index]);
}

static void training_target_text_by_index(int index, char *out, size_t out_len)
{
    if (!out || out_len == 0) {
        return;
    }
    const training_profile_t *profile = training_profile_by_index(index);
    if (profile->counted) {
        snprintf(out,
                 out_len,
                 "%d组 x %d次",
                 training_goal_sets_for_index(index),
                 training_goal_reps_for_index(index));
    } else {
        snprintf(out, out_len, "%u秒保持", (unsigned)training_goal_duration_for_index(index));
    }
}

static void load_training_goals(void)
{
    training_goal_set_defaults();

    nvs_handle_t h;
    if (nvs_open("train_goal", NVS_READONLY, &h) != ESP_OK) {
        return;
    }

    uint8_t version = 0;
    if (nvs_get_u8(h, "ver", &version) != ESP_OK || version != TRAINING_GOAL_VERSION) {
        nvs_close(h);
        return;
    }

    for (int i = 0; i < TRAINING_PROFILE_COUNT; ++i) {
        char key[8];
        uint8_t u8 = 0;
        snprintf(key, sizeof(key), "s%d", i);
        if (nvs_get_u8(h, key, &u8) == ESP_OK && u8 >= 1 && u8 <= 9) {
            s_training_goal_sets[i] = u8;
        }
        snprintf(key, sizeof(key), "r%d", i);
        if (nvs_get_u8(h, key, &u8) == ESP_OK && u8 >= 1 && u8 <= 99) {
            s_training_goal_reps[i] = u8;
        }
        uint16_t u16 = 0;
        snprintf(key, sizeof(key), "d%d", i);
        if (nvs_get_u16(h, key, &u16) == ESP_OK && u16 >= 15 && u16 <= 1800) {
            s_training_goal_duration_sec[i] = u16;
        }
    }
    nvs_close(h);
}

static esp_err_t save_training_goals(void)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open("train_goal", NVS_READWRITE, &h);
    if (err != ESP_OK) {
        return err;
    }

    for (int i = 0; i < TRAINING_PROFILE_COUNT && err == ESP_OK; ++i) {
        char key[8];
        if (i == 0) {
            err = nvs_set_u8(h, "ver", TRAINING_GOAL_VERSION);
            if (err != ESP_OK) {
                break;
            }
        }
        snprintf(key, sizeof(key), "s%d", i);
        err = nvs_set_u8(h, key, s_training_goal_sets[i]);
        if (err != ESP_OK) {
            break;
        }
        snprintf(key, sizeof(key), "r%d", i);
        err = nvs_set_u8(h, key, s_training_goal_reps[i]);
        if (err != ESP_OK) {
            break;
        }
        snprintf(key, sizeof(key), "d%d", i);
        err = nvs_set_u16(h, key, s_training_goal_duration_sec[i]);
    }
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    nvs_close(h);
    return err;
}

static int training_target_count(const training_profile_t *profile)
{
    if (!profile || !profile->counted) {
        return 0;
    }
    const int index = training_profile_index(profile);
    return std::max(1, training_goal_sets_for_index(index) * training_goal_reps_for_index(index));
}

static const char *training_state_text(TrainingRunState state)
{
    switch (state) {
    case TrainingRunState::Running:
        return "训练中";
    case TrainingRunState::Paused:
        return "已暂停";
    case TrainingRunState::Finished:
        return "已完成";
    case TrainingRunState::Ready:
    default:
        return "准备";
    }
}

static uint32_t training_elapsed_ms_locked(int64_t now_us)
{
    int64_t elapsed = s_training_elapsed_us;
    if (s_training_state == TrainingRunState::Running && s_training_start_us > 0) {
        elapsed += now_us - s_training_start_us;
    }
    if (elapsed < 0) {
        elapsed = 0;
    }
    return (uint32_t)(elapsed / 1000);
}

static void training_set_cue_locked(const char *cue)
{
    strlcpy(s_training_cue, cue ? cue : "", sizeof(s_training_cue));
}

static void training_rep_eval_reset(squat_rep_eval_t *rep);

static void training_reset_locked(void)
{
    const training_profile_t *profile = training_profile_by_index(s_training_index);
    s_training_state = TrainingRunState::Ready;
    s_training_count = 0;
    s_training_current_set = 1;
    s_training_start_us = 0;
    s_training_elapsed_us = 0;
    s_training_phase_active = false;
    s_training_stand_frames = 0;
    s_training_down_frames = 0;
    s_training_up_frames = 0;
    s_training_rep_cooldown_frames = 0;
    s_training_last_rep_us = 0;
    s_training_phase_start_us = 0;
    s_training_needs_rearm = false;
    s_training_motion_ref = 0.0f;
    s_training_phase_peak_signal = 0.0f;
    s_training_hold_good_frames = 0;
    s_training_hold_bad_frames = 0;
    s_training_score = 0;
    s_training_score_depth = 0;
    s_training_score_knee = 0;
    s_training_score_hip = 0;
    s_training_score_torso = 0;
    s_training_score_balance = 0;
    s_training_score_track = 0;
    s_training_session_score_sum = 0;
    s_training_session_depth_sum = 0;
    s_training_session_knee_sum = 0;
    s_training_session_hip_sum = 0;
    s_training_session_torso_sum = 0;
    s_training_session_balance_sum = 0;
    s_training_session_track_sum = 0;
    s_training_scored_reps = 0;
    s_training_good_rep_streak = 0;
    training_rep_eval_reset(&s_training_current_rep);
    training_rep_eval_reset(&s_training_last_rep);
    s_training_simple_rep_valid = false;
    s_training_simple_rep_depth = 0;
    s_training_simple_rep_knee = 0;
    s_training_simple_rep_hip = 0;
    s_training_simple_rep_torso = 0;
    s_training_simple_rep_balance = 0;
    s_training_simple_rep_track = 0;
    s_training_knee_angle = 0.0f;
    s_training_hip_angle = 0.0f;
    s_training_torso_lean = 0.0f;
    s_training_depth_ratio = 0.0f;
    s_training_symmetry_delta = 0.0f;
    s_training_valid_kpts = 0;
    s_training_pc_fps = 0.0f;
    s_training_last_pose_us = 0;
    s_training_last_debug_us = 0;
    s_training_last_voice_prompt = VoicePromptId::TrainingComplete;
    s_training_last_voice_us = 0;
    s_training_record_saved = false;
    char cue[120];
    snprintf(cue, sizeof(cue), "已选择%s，点击开始进入训练。", profile->name);
    training_set_cue_locked(cue);
    strlcpy(s_training_detail, "开始训练后显示分项评分。", sizeof(s_training_detail));
}

static training_snapshot_t training_get_snapshot(void)
{
    training_snapshot_t snap = {};
    const int64_t now_us = esp_timer_get_time();
    training_lock();
    snap.profile = training_profile_by_index(s_training_index);
    snap.state = s_training_state;
    snap.count = s_training_count;
    snap.target_count = training_target_count(snap.profile);
    snap.current_set = s_training_current_set;
    snap.elapsed_ms = training_elapsed_ms_locked(now_us);
    snap.score = s_training_score;
    snap.score_depth = s_training_score_depth;
    snap.score_knee = s_training_score_knee;
    snap.score_hip = s_training_score_hip;
    snap.score_torso = s_training_score_torso;
    snap.score_balance = s_training_score_balance;
    snap.score_track = s_training_score_track;
    snap.knee_angle = s_training_knee_angle;
    snap.hip_angle = s_training_hip_angle;
    snap.torso_lean = s_training_torso_lean;
    snap.depth_ratio = s_training_depth_ratio;
    snap.symmetry_delta = s_training_symmetry_delta;
    snap.valid_kpts = s_training_valid_kpts;
    snap.pc_fps = s_training_pc_fps;
    strlcpy(snap.cue, s_training_cue, sizeof(snap.cue));
    strlcpy(snap.detail, s_training_detail, sizeof(snap.detail));
    training_unlock();
    return snap;
}

static void format_record_time(char *out, size_t out_len)
{
    if (!out || out_len == 0) {
        return;
    }
    time_t now = time(nullptr);
    struct tm tm_now = {};
    localtime_r(&now, &tm_now);
    if (tm_now.tm_year < 120) {
        strlcpy(out, "--", out_len);
        return;
    }
    snprintf(out,
             out_len,
             "%04d-%02d-%02d %02d:%02d",
             tm_now.tm_year + 1900,
             tm_now.tm_mon + 1,
             tm_now.tm_mday,
             tm_now.tm_hour,
             tm_now.tm_min);
}

static void training_record_save_snapshot(const training_snapshot_t &snap)
{
    if (!snap.profile || (snap.elapsed_ms == 0 && snap.count == 0)) {
        return;
    }
    s_training_record.sessions++;
    s_training_record.last_score = snap.score;
    s_training_record.last_score_depth = snap.score_depth;
    s_training_record.last_score_knee = snap.score_knee;
    s_training_record.last_score_hip = snap.score_hip;
    s_training_record.last_score_torso = snap.score_torso;
    s_training_record.last_score_balance = snap.score_balance;
    s_training_record.last_score_track = snap.score_track;
    s_training_record.last_count = snap.count;
    s_training_record.last_target = snap.target_count;
    s_training_record.last_elapsed_ms = snap.elapsed_ms;
    s_training_record.total_count += (uint32_t)std::max(0, snap.count);
    s_training_record.total_elapsed_ms += snap.elapsed_ms;
    strlcpy(s_training_record.last_training, snap.profile->name, sizeof(s_training_record.last_training));
    strlcpy(s_training_record.last_tip,
            snap.detail[0] ? snap.detail : (snap.cue[0] ? snap.cue : "训练已完成。"),
            sizeof(s_training_record.last_tip));
    format_record_time(s_training_record.last_time, sizeof(s_training_record.last_time));
    esp_err_t err = save_training_record_to_nvs();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "save training record failed: %s", esp_err_to_name(err));
    }
    training_history_item_t item = {};
    item.valid = true;
    strlcpy(item.source, "live", sizeof(item.source));
    strlcpy(item.action, snap.profile->name, sizeof(item.action));
    strlcpy(item.time, s_training_record.last_time, sizeof(item.time));
    strlcpy(item.tip, s_training_record.last_tip, sizeof(item.tip));
    item.count = snap.count;
    item.target = snap.target_count;
    item.score = snap.score;
    item.rep_score = snap.score;
    item.score_depth = snap.score_depth;
    item.score_knee = snap.score_knee;
    item.score_hip = snap.score_hip;
    item.score_torso = snap.score_torso;
    item.score_balance = snap.score_balance;
    item.score_track = snap.score_track;
    item.elapsed_ms = snap.elapsed_ms;
    training_history_append(item);
}

static void training_record_save_current_locked(const char *fallback_tip)
{
    if (s_training_record_saved) {
        return;
    }
    const int64_t now_us = esp_timer_get_time();
    training_snapshot_t snap = {};
    snap.profile = training_profile_by_index(s_training_index);
    snap.state = s_training_state;
    snap.count = s_training_count;
    snap.target_count = training_target_count(snap.profile);
    snap.current_set = s_training_current_set;
    snap.elapsed_ms = training_elapsed_ms_locked(now_us);
    snap.score = s_training_score;
    snap.score_depth = s_training_score_depth;
    snap.score_knee = s_training_score_knee;
    snap.score_hip = s_training_score_hip;
    snap.score_torso = s_training_score_torso;
    snap.score_balance = s_training_score_balance;
    snap.score_track = s_training_score_track;
    strlcpy(snap.cue, s_training_cue[0] ? s_training_cue : (fallback_tip ? fallback_tip : "训练已结束。"), sizeof(snap.cue));
    strlcpy(snap.detail, s_training_detail, sizeof(snap.detail));
    if (snap.elapsed_ms == 0 && snap.count == 0) {
        return;
    }
    s_training_record_saved = true;
    training_record_save_snapshot(snap);
}

static void training_select_index(int index)
{
    training_lock();
    if (index < 0 || index >= TRAINING_PROFILE_COUNT) {
        index = 0;
    }
    s_training_index = index;
    training_reset_locked();
    training_unlock();
}

static void training_start_or_resume(void)
{
    training_lock();
    const int64_t now_us = esp_timer_get_time();
    if (s_training_state == TrainingRunState::Finished) {
        training_reset_locked();
    }
    if (s_training_state == TrainingRunState::Ready || s_training_state == TrainingRunState::Paused) {
        s_training_state = TrainingRunState::Running;
        s_training_start_us = now_us;
        s_training_phase_active = false;
        s_training_stand_frames = 0;
        s_training_down_frames = 0;
        s_training_up_frames = 0;
        s_training_rep_cooldown_frames = 0;
        s_training_last_rep_us = 0;
        s_training_phase_start_us = 0;
        s_training_needs_rearm = false;
        s_training_motion_ref = 0.0f;
        s_training_phase_peak_signal = 0.0f;
        s_training_hold_good_frames = 0;
        s_training_hold_bad_frames = 0;
        training_rep_eval_reset(&s_training_current_rep);
        const training_profile_t *profile = training_profile_by_index(s_training_index);
        char cue[120];
        if (s_training_index == 5) {
            s_training_start_us = 0;
            snprintf(cue, sizeof(cue), "%s开始，进入支撑姿态后自动计时。", profile->name);
        } else {
            snprintf(cue, sizeof(cue), "%s开始，跟随绿色PC骨架完成动作。", profile->name);
        }
        training_set_cue_locked(cue);
    }
    training_unlock();
}

static void training_pause_if_running(void)
{
    training_lock();
    if (s_training_state == TrainingRunState::Running) {
        const int64_t now_us = esp_timer_get_time();
        if (s_training_start_us > 0) {
            s_training_elapsed_us += now_us - s_training_start_us;
        }
        s_training_start_us = 0;
        s_training_state = TrainingRunState::Paused;
        training_set_cue_locked("训练已暂停，返回实时训练后可以继续。");
    }
    training_unlock();
}

static void training_finish_session(void)
{
    training_lock();
    if (s_training_state == TrainingRunState::Running && s_training_start_us > 0) {
        s_training_elapsed_us += esp_timer_get_time() - s_training_start_us;
    }
    s_training_start_us = 0;
    s_training_state = TrainingRunState::Finished;
    s_training_phase_active = false;
    s_training_stand_frames = 0;
    s_training_down_frames = 0;
    s_training_up_frames = 0;
    s_training_rep_cooldown_frames = 0;
    s_training_last_rep_us = 0;
    s_training_phase_start_us = 0;
    s_training_motion_ref = 0.0f;
    s_training_phase_peak_signal = 0.0f;
    s_training_hold_good_frames = 0;
    s_training_hold_bad_frames = 0;
    training_set_cue_locked("训练已结束，可以查看报告或重新开始。");
    training_record_save_current_locked("训练已结束，可以查看报告或重新开始。");
    training_unlock();
}

static void training_toggle_session(void)
{
    training_lock();
    const TrainingRunState state = s_training_state;
    training_unlock();

    if (state == TrainingRunState::Running) {
        training_pause_if_running();
    } else {
        training_start_or_resume();
    }
}

static bool training_get_keypoint(const pose_result_t &pose, int index, float min_score, pose_keypoint_t *out)
{
    if (index < 0 || index >= (int)pose.keypoints.size()) {
        return false;
    }
    const pose_keypoint_t &kp = pose.keypoints[index];
    if (kp.score < min_score) {
        return false;
    }
    if (out) {
        *out = kp;
    }
    return true;
}

static bool training_avg_y(const pose_result_t &pose, int a, int b, float min_score, float *out_y)
{
    pose_keypoint_t ka = {};
    pose_keypoint_t kb = {};
    int n = 0;
    float sum = 0.0f;
    if (training_get_keypoint(pose, a, min_score, &ka)) {
        sum += (float)ka.y;
        n++;
    }
    if (training_get_keypoint(pose, b, min_score, &kb)) {
        sum += (float)kb.y;
        n++;
    }
    if (n == 0) {
        return false;
    }
    *out_y = sum / (float)n;
    return true;
}

static bool training_avg_keypoint(const pose_result_t &pose,
                                  int a,
                                  int b,
                                  float min_score,
                                  pose_keypoint_t *out)
{
    pose_keypoint_t ka = {};
    pose_keypoint_t kb = {};
    int n = 0;
    float x = 0.0f;
    float y = 0.0f;
    float score = 0.0f;
    if (training_get_keypoint(pose, a, min_score, &ka)) {
        x += (float)ka.x;
        y += (float)ka.y;
        score += ka.score;
        n++;
    }
    if (training_get_keypoint(pose, b, min_score, &kb)) {
        x += (float)kb.x;
        y += (float)kb.y;
        score += kb.score;
        n++;
    }
    if (n == 0 || !out) {
        return false;
    }
    out->x = (int)lroundf(x / (float)n);
    out->y = (int)lroundf(y / (float)n);
    out->score = score / (float)n;
    return true;
}

static void training_add_rep_locked(const squat_rep_eval_t *rep_eval = nullptr);

typedef struct {
    bool valid;
    float confidence;
    float knee_angle;
    bool knee_angle_valid;
    float knee_track_delta;
    bool knee_track_valid;
    float hip_angle;
    float torso_lean;
    float depth;
    bool both_sides;
    float symmetry_delta;
} squat_metrics_t;

static float training_clampf(float v, float lo, float hi)
{
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static float training_angle_deg(const pose_keypoint_t &a, const pose_keypoint_t &b, const pose_keypoint_t &c)
{
    const float abx = (float)a.x - (float)b.x;
    const float aby = (float)a.y - (float)b.y;
    const float cbx = (float)c.x - (float)b.x;
    const float cby = (float)c.y - (float)b.y;
    const float ab_len = sqrtf(abx * abx + aby * aby);
    const float cb_len = sqrtf(cbx * cbx + cby * cby);
    if (ab_len < 1.0f || cb_len < 1.0f) {
        return 180.0f;
    }
    const float cos_v = training_clampf((abx * cbx + aby * cby) / (ab_len * cb_len), -1.0f, 1.0f);
    return acosf(cos_v) * 57.2957795f;
}

static int training_score_from01(float v)
{
    return std::max(0, std::min(100, (int)lroundf(training_clampf(v, 0.0f, 1.0f) * 100.0f)));
}

static void training_rep_eval_reset(squat_rep_eval_t *rep)
{
    if (!rep) {
        return;
    }
    *rep = {};
    rep->max_depth = -1.0f;
    rep->min_knee_angle = 999.0f;
    rep->max_knee_track_delta = 0.0f;
    rep->min_hip_angle = 999.0f;
    rep->voice = VoicePromptId::BadDepth;
    strlcpy(rep->detail, "动作开始，系统将按本次完整动作评分。", sizeof(rep->detail));
}

static void training_rep_eval_add_sample(squat_rep_eval_t *rep,
                                         const squat_metrics_t &m,
                                         int valid_count)
{
    if (!rep) {
        return;
    }
    rep->samples++;
    rep->confidence_sum += m.confidence;
    rep->valid_kpts_sum += valid_count;
    rep->max_depth = std::max(rep->max_depth, m.depth);
    rep->min_hip_angle = std::min(rep->min_hip_angle, m.hip_angle);
    rep->max_torso_lean = std::max(rep->max_torso_lean, m.torso_lean);
    rep->max_symmetry_delta = std::max(rep->max_symmetry_delta, m.symmetry_delta);
    if (m.knee_track_valid) {
        rep->has_knee_track = true;
        rep->max_knee_track_delta = std::max(rep->max_knee_track_delta, m.knee_track_delta);
    }
    if (m.knee_angle_valid) {
        rep->has_knee_angle = true;
        rep->min_knee_angle = std::min(rep->min_knee_angle, m.knee_angle);
    }
}

static void training_rep_eval_finalize(squat_rep_eval_t *rep)
{
    if (!rep || rep->samples <= 0) {
        return;
    }

    const float avg_conf = rep->confidence_sum / (float)rep->samples;
    const float avg_valid = (float)rep->valid_kpts_sum / (float)rep->samples;
    const int depth_ratio_score = training_score_from01((rep->max_depth - 0.52f) / 0.18f);
    int depth_angle_score = 0;
    bool has_depth_angle = false;
    if (rep->has_knee_angle) {
        depth_angle_score = std::max(depth_angle_score, training_score_from01((142.0f - rep->min_knee_angle) / 42.0f));
        has_depth_angle = true;
    }
    if (rep->min_hip_angle < 998.0f) {
        depth_angle_score = std::max(depth_angle_score, training_score_from01((158.0f - rep->min_hip_angle) / 58.0f));
        has_depth_angle = true;
    }
    rep->score_depth = has_depth_angle ? std::max(depth_ratio_score, depth_angle_score) : depth_ratio_score;
    const int knee_angle_score = rep->has_knee_angle ?
                                 training_score_from01(1.0f - std::max(0.0f,
                                                                        std::max(82.0f - rep->min_knee_angle,
                                                                                 rep->min_knee_angle - 142.0f)) /
                                                                       48.0f) :
                                 72;
    const int knee_track_score = rep->has_knee_track ?
                                 training_score_from01(1.0f - (rep->max_knee_track_delta - 0.22f) / 0.34f) :
                                 74;
    rep->score_knee = std::max(0,
                               std::min(100,
                                        (int)lroundf((float)knee_track_score * 0.65f +
                                                     (float)knee_angle_score * 0.35f)));
    rep->score_hip = training_score_from01(1.0f - fabsf(rep->min_hip_angle - 98.0f) / 72.0f);
    rep->score_torso = training_score_from01(1.0f - (rep->max_torso_lean - 28.0f) / 28.0f);
    rep->score_balance = rep->max_symmetry_delta > 0.0f ? training_score_from01(1.0f - (rep->max_symmetry_delta - 16.0f) / 34.0f) : 78;
    const float track = training_clampf((avg_conf - 0.12f) / 0.42f, 0.0f, 1.0f) * 0.55f +
                        training_clampf(avg_valid / 17.0f, 0.0f, 1.0f) * 0.45f;
    rep->score_track = training_score_from01(track);
    rep->score = std::max(0,
                          std::min(100,
                                   (int)lroundf((float)rep->score_depth * 0.30f +
                                                (float)rep->score_knee * 0.20f +
                                                (float)rep->score_hip * 0.15f +
                                                (float)rep->score_torso * 0.15f +
                                                (float)rep->score_balance * 0.10f +
                                                (float)rep->score_track * 0.10f)));

    rep->needs_voice = false;
    rep->good = rep->score >= 85 &&
                rep->score_depth >= 78 &&
                rep->score_torso >= 70 &&
                rep->score_knee >= 62;
    if (rep->score_depth < 48) {
        rep->voice = VoicePromptId::BadDepth;
        rep->needs_voice = true;
        strlcpy(rep->detail, "本次下蹲深度不足，髋部继续向后向下。", sizeof(rep->detail));
    } else if (rep->score_torso < 50) {
        rep->voice = VoicePromptId::BadTorso;
        rep->needs_voice = true;
        strlcpy(rep->detail, "本次躯干前倾偏大，起身时抬胸并收紧核心。", sizeof(rep->detail));
    } else if (rep->score_balance < 46) {
        rep->voice = VoicePromptId::BadBalance;
        rep->needs_voice = true;
        strlcpy(rep->detail, "本次左右发力不够均衡，注意两侧同时起身。", sizeof(rep->detail));
    } else if (rep->score_knee < 44) {
        rep->voice = VoicePromptId::BadKnee;
        rep->needs_voice = true;
        strlcpy(rep->detail, "本次膝盖轨迹偏移，下一次让膝盖跟随脚尖方向。", sizeof(rep->detail));
    } else if (rep->score < 65) {
        rep->voice = VoicePromptId::BadDepth;
        rep->needs_voice = true;
        strlcpy(rep->detail, "本次动作质量偏低，下一次放慢节奏并做完整。", sizeof(rep->detail));
    } else {
        strlcpy(rep->detail, "本次动作完成，继续保持稳定节奏。", sizeof(rep->detail));
    }
}

static void training_apply_rep_score_locked(const squat_rep_eval_t &rep)
{
    if (rep.samples <= 0) {
        return;
    }
    s_training_last_rep = rep;
    s_training_session_score_sum += rep.score;
    s_training_session_depth_sum += rep.score_depth;
    s_training_session_knee_sum += rep.score_knee;
    s_training_session_hip_sum += rep.score_hip;
    s_training_session_torso_sum += rep.score_torso;
    s_training_session_balance_sum += rep.score_balance;
    s_training_session_track_sum += rep.score_track;
    s_training_scored_reps++;

    const int n = std::max(1, s_training_scored_reps);
    s_training_score = s_training_session_score_sum / n;
    s_training_score_depth = s_training_session_depth_sum / n;
    s_training_score_knee = s_training_session_knee_sum / n;
    s_training_score_hip = s_training_session_hip_sum / n;
    s_training_score_torso = s_training_session_torso_sum / n;
    s_training_score_balance = s_training_session_balance_sum / n;
    s_training_score_track = s_training_session_track_sum / n;
    strlcpy(s_training_detail, rep.detail, sizeof(s_training_detail));

    if (rep.good) {
        s_training_good_rep_streak++;
    } else {
        s_training_good_rep_streak = 0;
    }
}

static void training_set_live_scores_locked(int total,
                                            int depth,
                                            int knee,
                                            int hip,
                                            int torso,
                                            int balance,
                                            int track,
                                            const char *detail)
{
    s_training_score = std::max(0, std::min(100, total));
    s_training_score_depth = std::max(0, std::min(100, depth));
    s_training_score_knee = std::max(0, std::min(100, knee));
    s_training_score_hip = std::max(0, std::min(100, hip));
    s_training_score_torso = std::max(0, std::min(100, torso));
    s_training_score_balance = std::max(0, std::min(100, balance));
    s_training_score_track = std::max(0, std::min(100, track));
    if (detail) {
        strlcpy(s_training_detail, detail, sizeof(s_training_detail));
    }
}

static void training_commit_simple_rep_locked(int depth,
                                              int knee,
                                              int hip,
                                              int torso,
                                              int balance,
                                              int track,
                                              VoicePromptId voice,
                                              bool needs_voice,
                                              const char *detail)
{
    squat_rep_eval_t rep = {};
    rep.samples = 1;
    rep.score_depth = std::max(0, std::min(100, depth));
    rep.score_knee = std::max(0, std::min(100, knee));
    rep.score_hip = std::max(0, std::min(100, hip));
    rep.score_torso = std::max(0, std::min(100, torso));
    rep.score_balance = std::max(0, std::min(100, balance));
    rep.score_track = std::max(0, std::min(100, track));
    rep.score = std::max(0,
                         std::min(100,
                                  (int)lroundf((float)rep.score_depth * 0.28f +
                                               (float)rep.score_knee * 0.16f +
                                               (float)rep.score_hip * 0.16f +
                                               (float)rep.score_torso * 0.18f +
                                               (float)rep.score_balance * 0.10f +
                                               (float)rep.score_track * 0.12f)));
    rep.voice = voice;
    rep.needs_voice = needs_voice;
    rep.good = rep.score >= 84 && !needs_voice;
    strlcpy(rep.detail, detail ? detail : "本次动作完成。", sizeof(rep.detail));
    training_apply_rep_score_locked(rep);
    training_add_rep_locked(&rep);
}

static int training_clamp_score(int score)
{
    return std::max(0, std::min(100, score));
}

static void training_simple_rep_reset_locked(int depth,
                                             int knee,
                                             int hip,
                                             int torso,
                                             int balance,
                                             int track)
{
    s_training_simple_rep_valid = true;
    s_training_simple_rep_depth = training_clamp_score(depth);
    s_training_simple_rep_knee = training_clamp_score(knee);
    s_training_simple_rep_hip = training_clamp_score(hip);
    s_training_simple_rep_torso = training_clamp_score(torso);
    s_training_simple_rep_balance = training_clamp_score(balance);
    s_training_simple_rep_track = training_clamp_score(track);
}

static void training_simple_rep_sample_locked(int depth,
                                              int knee,
                                              int hip,
                                              int torso,
                                              int balance,
                                              int track)
{
    if (!s_training_simple_rep_valid) {
        training_simple_rep_reset_locked(depth, knee, hip, torso, balance, track);
        return;
    }
    s_training_simple_rep_depth = std::max(s_training_simple_rep_depth, training_clamp_score(depth));
    s_training_simple_rep_knee = std::max(s_training_simple_rep_knee, training_clamp_score(knee));
    s_training_simple_rep_hip = std::max(s_training_simple_rep_hip, training_clamp_score(hip));
    s_training_simple_rep_torso = std::max(s_training_simple_rep_torso, training_clamp_score(torso));
    s_training_simple_rep_balance = std::max(s_training_simple_rep_balance, training_clamp_score(balance));
    s_training_simple_rep_track = std::max(s_training_simple_rep_track, training_clamp_score(track));
}

static void training_commit_simple_rep_best_locked(int depth,
                                                   int knee,
                                                   int hip,
                                                   int torso,
                                                   int balance,
                                                   int track,
                                                   VoicePromptId voice,
                                                   bool needs_voice,
                                                   const char *detail)
{
    training_simple_rep_sample_locked(depth, knee, hip, torso, balance, track);
    const int best_depth = s_training_simple_rep_valid ? s_training_simple_rep_depth : depth;
    const int best_knee = s_training_simple_rep_valid ? s_training_simple_rep_knee : knee;
    const int best_hip = s_training_simple_rep_valid ? s_training_simple_rep_hip : hip;
    const int best_torso = s_training_simple_rep_valid ? s_training_simple_rep_torso : torso;
    const int best_balance = s_training_simple_rep_valid ? s_training_simple_rep_balance : balance;
    const int best_track = s_training_simple_rep_valid ? s_training_simple_rep_track : track;
    s_training_simple_rep_valid = false;
    training_commit_simple_rep_locked(best_depth,
                                      best_knee,
                                      best_hip,
                                      best_torso,
                                      best_balance,
                                      best_track,
                                      voice,
                                      needs_voice,
                                      detail);
}

static int training_track_score(int valid_count, float avg_score)
{
    const float by_score = training_clampf((avg_score - 0.12f) / 0.42f, 0.0f, 1.0f);
    const float by_count = training_clampf((float)valid_count / 17.0f, 0.0f, 1.0f);
    return training_score_from01(by_score * 0.55f + by_count * 0.45f);
}

static int training_range_score(float value, float good_lo, float good_hi, float tolerance)
{
    if (value >= good_lo && value <= good_hi) {
        return 100;
    }
    const float miss = value < good_lo ? good_lo - value : value - good_hi;
    return training_score_from01(1.0f - miss / std::max(1.0f, tolerance));
}

static int64_t training_pose_now_us(void)
{
    return s_training_last_pose_us > 0 ? s_training_last_pose_us : esp_timer_get_time();
}

static void training_tick_rep_cooldown_locked(void)
{
    if (s_training_rep_cooldown_frames > 0) {
        s_training_rep_cooldown_frames--;
    }
}

static int64_t training_min_rep_gap_us(void)
{
    switch (s_training_index) {
    case 0: // 深蹲
        return 2200000;
    case 1: // 俯卧撑
        return 1200000;
    case 2: // 卧推
        return 1200000;
    case 3: // 引体向上
        return 2000000;
    case 4: // 硬拉
        return 1800000;
    case 6: // 哑铃弯举
        return 1150000;
    case 7: // 仰卧起坐
        return 1400000;
    default:
        return 1400000;
    }
}

static bool training_rep_gap_ready_locked(void)
{
    const int64_t now_us = training_pose_now_us();
    return s_training_last_rep_us == 0 || now_us - s_training_last_rep_us >= training_min_rep_gap_us();
}

static void training_clear_rearm_when_ready_locked(bool ready_condition, int ready_frames)
{
    if (ready_condition) {
        s_training_stand_frames = std::min(s_training_stand_frames + 1, 12);
        if (s_training_needs_rearm && s_training_stand_frames >= ready_frames) {
            s_training_needs_rearm = false;
        }
    } else {
        s_training_stand_frames = std::max(0, s_training_stand_frames - 1);
    }
}

static bool training_can_start_rep_locked(int ready_frames)
{
    return !s_training_needs_rearm &&
           s_training_rep_cooldown_frames <= 0 &&
           training_rep_gap_ready_locked() &&
           s_training_stand_frames >= ready_frames;
}

static bool training_squat_side_metrics(const pose_result_t &pose, bool left, squat_metrics_t *out)
{
    if (!out) {
        return false;
    }
    *out = {};

    const int shoulder_idx = left ? POSE_KPT_LEFT_SHOULDER : POSE_KPT_RIGHT_SHOULDER;
    const int alt_shoulder_idx = left ? POSE_KPT_RIGHT_SHOULDER : POSE_KPT_LEFT_SHOULDER;
    const int hip_idx = left ? POSE_KPT_LEFT_HIP : POSE_KPT_RIGHT_HIP;
    const int knee_idx = left ? POSE_KPT_LEFT_KNEE : POSE_KPT_RIGHT_KNEE;
    const int ankle_idx = left ? POSE_KPT_LEFT_ANKLE : POSE_KPT_RIGHT_ANKLE;

    pose_keypoint_t shoulder = {};
    pose_keypoint_t hip = {};
    pose_keypoint_t knee = {};
    pose_keypoint_t ankle = {};
    if (!training_get_keypoint(pose, shoulder_idx, 0.12f, &shoulder)) {
        if (!training_get_keypoint(pose, alt_shoulder_idx, 0.12f, &shoulder)) {
            return false;
        }
    }
    if (!training_get_keypoint(pose, hip_idx, 0.10f, &hip) ||
        !training_get_keypoint(pose, knee_idx, 0.10f, &knee)) {
        return false;
    }
    const bool ankle_ok = training_get_keypoint(pose, ankle_idx, 0.08f, &ankle);

    out->valid = true;
    out->confidence = ankle_ok ? (shoulder.score + hip.score + knee.score + ankle.score) * 0.25f :
                                 (shoulder.score + hip.score + knee.score) * 0.3333333f;
    out->knee_angle_valid = ankle_ok;
    out->knee_angle = ankle_ok ? training_angle_deg(hip, knee, ankle) : 180.0f;
    if (ankle_ok) {
        const float leg_dx = (float)knee.x - (float)ankle.x;
        const float leg_dy = (float)knee.y - (float)ankle.y;
        const float leg_len = std::max(32.0f, sqrtf(leg_dx * leg_dx + leg_dy * leg_dy));
        out->knee_track_delta = fabsf((float)knee.x - (float)ankle.x) / leg_len;
        out->knee_track_valid = true;
    }
    out->hip_angle = training_angle_deg(shoulder, hip, knee);
    const float torso_dx = fabsf((float)shoulder.x - (float)hip.x);
    const float torso_dy = fabsf((float)shoulder.y - (float)hip.y);
    out->torso_lean = atan2f(torso_dx, std::max(1.0f, torso_dy)) * 57.2957795f;
    const float denom = (float)knee.y - (float)shoulder.y;
    out->depth = denom > 20.0f ? training_clampf(((float)hip.y - (float)shoulder.y) / denom, -0.25f, 1.35f) : -1.0f;
    return out->depth >= -0.05f;
}

static bool training_squat_metrics(const pose_result_t &pose, squat_metrics_t *out)
{
    if (!out) {
        return false;
    }
    squat_metrics_t left = {};
    squat_metrics_t right = {};
    const bool left_ok = training_squat_side_metrics(pose, true, &left);
    const bool right_ok = training_squat_side_metrics(pose, false, &right);
    if (!left_ok && !right_ok) {
        *out = {};
        return false;
    }
    if (left_ok && !right_ok) {
        *out = left;
        out->symmetry_delta = 0.0f;
        return true;
    }
    if (!left_ok && right_ok) {
        *out = right;
        out->symmetry_delta = 0.0f;
        return true;
    }

    const float lw = std::max(0.05f, left.confidence);
    const float rw = std::max(0.05f, right.confidence);
    const float sum = lw + rw;
    out->valid = true;
    out->both_sides = true;
    out->confidence = (left.confidence * lw + right.confidence * rw) / sum;
    float knee_sum = 0.0f;
    float knee_w = 0.0f;
    if (left.knee_angle_valid) {
        knee_sum += left.knee_angle * lw;
        knee_w += lw;
    }
    if (right.knee_angle_valid) {
        knee_sum += right.knee_angle * rw;
        knee_w += rw;
    }
    out->knee_angle_valid = knee_w > 0.0f;
    out->knee_angle = out->knee_angle_valid ? knee_sum / knee_w : 180.0f;
    float track_sum = 0.0f;
    float track_w = 0.0f;
    if (left.knee_track_valid) {
        track_sum += left.knee_track_delta * lw;
        track_w += lw;
    }
    if (right.knee_track_valid) {
        track_sum += right.knee_track_delta * rw;
        track_w += rw;
    }
    out->knee_track_valid = track_w > 0.0f;
    out->knee_track_delta = out->knee_track_valid ? track_sum / track_w : 0.0f;
    out->hip_angle = (left.hip_angle * lw + right.hip_angle * rw) / sum;
    out->torso_lean = (left.torso_lean * lw + right.torso_lean * rw) / sum;
    out->depth = (left.depth * lw + right.depth * rw) / sum;
    out->symmetry_delta = (left.knee_angle_valid && right.knee_angle_valid) ? fabsf(left.knee_angle - right.knee_angle) : 0.0f;
    return true;
}

static void training_update_squat_locked(const pose_result_t &pose, int valid_count, float avg_score)
{
    squat_metrics_t m = {};
    if (!training_squat_metrics(pose, &m)) {
        s_training_stand_frames = 0;
        s_training_down_frames = 0;
        s_training_up_frames = 0;
        if (s_training_scored_reps == 0) {
            s_training_score = 0;
            s_training_score_depth = 0;
            s_training_score_knee = 0;
            s_training_score_hip = 0;
            s_training_score_torso = 0;
            s_training_score_balance = 0;
            s_training_score_track = std::max(0, std::min(100, (int)lroundf(training_clampf((float)valid_count / 17.0f, 0.0f, 1.0f) * 100.0f)));
        }
        s_training_knee_angle = 0.0f;
        s_training_hip_angle = 0.0f;
        s_training_torso_lean = 0.0f;
        s_training_depth_ratio = 0.0f;
        s_training_symmetry_delta = 0.0f;
        training_set_cue_locked("请让肩、髋、膝、踝都进入画面。");
        strlcpy(s_training_detail, "骨架关键点不足，先调整站位和摄像头视角。", sizeof(s_training_detail));
        return;
    }

    const int64_t now_train_us = s_training_last_pose_us > 0 ? s_training_last_pose_us : esp_timer_get_time();
    const bool knee_straight = !m.knee_angle_valid || m.knee_angle >= 148.0f;
    const bool knee_bent = !m.knee_angle_valid || m.knee_angle <= 155.0f;
    const bool stand_condition = m.depth <= 0.62f && knee_straight && m.hip_angle >= 130.0f;
    const bool deep_by_angle = m.knee_angle_valid && m.knee_angle <= 132.0f && m.hip_angle <= 158.0f && m.depth >= 0.56f;
    const bool deep_by_depth = m.depth >= 0.66f && knee_bent && m.hip_angle <= 168.0f;
    const bool deep_by_strong_depth = m.depth >= 0.74f && m.hip_angle <= 174.0f;
    const bool down_condition = deep_by_angle || deep_by_depth;
    const bool bottom_condition = down_condition || deep_by_strong_depth;
    const bool up_condition = m.depth <= 0.58f && knee_straight && m.hip_angle >= 132.0f;
    const bool rep_gap_ok = training_rep_gap_ready_locked();
    const bool bottom_duration_ok = s_training_phase_start_us > 0 && now_train_us - s_training_phase_start_us >= 240000;
    if (s_training_rep_cooldown_frames > 0) {
        s_training_rep_cooldown_frames--;
    }

    const bool judging_bottom = s_training_phase_active || bottom_condition;
    const float track_score = training_clampf((m.confidence - 0.14f) / 0.42f, 0.0f, 1.0f) * 35.0f +
                              training_clampf((float)valid_count / 17.0f, 0.0f, 1.0f) * 10.0f;
    float depth_unit = training_clampf((m.depth - 0.48f) / 0.20f, 0.0f, 1.0f);
    float depth_angle_unit = training_clampf((158.0f - m.hip_angle) / 58.0f, 0.0f, 1.0f);
    if (m.knee_angle_valid) {
        depth_angle_unit = std::max(depth_angle_unit, training_clampf((142.0f - m.knee_angle) / 42.0f, 0.0f, 1.0f));
    }
    depth_unit = std::max(depth_unit, depth_angle_unit);
    const float depth_score = judging_bottom ? depth_unit * 15.0f : 10.0f;
    const float target_knee = judging_bottom ? 100.0f : 172.0f;
    const float target_hip = judging_bottom ? 92.0f : 172.0f;
    const float knee_angle_score = m.knee_angle_valid ?
                                   (1.0f - training_clampf(fabsf(m.knee_angle - target_knee) / 66.0f, 0.0f, 1.0f)) * 8.0f :
                                   4.5f;
    const float knee_track_score = m.knee_track_valid ?
                                   (1.0f - training_clampf((m.knee_track_delta - 0.22f) / 0.34f, 0.0f, 1.0f)) * 10.0f :
                                   7.0f;
    const float knee_score = knee_angle_score + knee_track_score;
    const float hip_score = (1.0f - training_clampf(fabsf(m.hip_angle - target_hip) / 70.0f, 0.0f, 1.0f)) * 10.0f;
    const float torso_score = (1.0f - training_clampf((m.torso_lean - 18.0f) / 35.0f, 0.0f, 1.0f)) * 15.0f;
    const float symmetry_score = m.both_sides ? (1.0f - training_clampf((m.symmetry_delta - 10.0f) / 35.0f, 0.0f, 1.0f)) * 7.0f : 4.0f;
    const int live_score = std::max(0, std::min(100, (int)lroundf(track_score + depth_score + knee_score + hip_score + torso_score + symmetry_score)));
    const int live_track = std::max(0, std::min(100, (int)lroundf(track_score / 45.0f * 100.0f)));
    const int live_depth = std::max(0, std::min(100, (int)lroundf(depth_score / 15.0f * 100.0f)));
    const int live_knee = std::max(0, std::min(100, (int)lroundf(knee_score / 18.0f * 100.0f)));
    const int live_hip = std::max(0, std::min(100, (int)lroundf(hip_score / 10.0f * 100.0f)));
    const int live_torso = std::max(0, std::min(100, (int)lroundf(torso_score / 15.0f * 100.0f)));
    const int live_balance = std::max(0, std::min(100, (int)lroundf(symmetry_score / 7.0f * 100.0f)));
    if (s_training_scored_reps == 0 || s_training_phase_active || bottom_condition) {
        s_training_score = live_score;
        s_training_score_track = live_track;
        s_training_score_depth = live_depth;
        s_training_score_knee = live_knee;
        s_training_score_hip = live_hip;
        s_training_score_torso = live_torso;
        s_training_score_balance = live_balance;
    }
    s_training_knee_angle = m.knee_angle;
    s_training_hip_angle = m.hip_angle;
    s_training_torso_lean = m.torso_lean;
    s_training_depth_ratio = m.depth;
    s_training_symmetry_delta = m.symmetry_delta;

    if (!s_training_phase_active) {
        s_training_up_frames = 0;
        if (stand_condition) {
            s_training_stand_frames = std::min(s_training_stand_frames + 1, 12);
            if (s_training_needs_rearm && s_training_stand_frames >= 4) {
                s_training_needs_rearm = false;
                training_rep_eval_reset(&s_training_current_rep);
            }
        } else if (!bottom_condition) {
            s_training_stand_frames = std::max(0, s_training_stand_frames - 1);
        }
        const bool ready_for_next_rep = !s_training_needs_rearm && (s_training_stand_frames >= 3 || s_training_count == 0);
        if (!rep_gap_ok || s_training_rep_cooldown_frames > 0 || !ready_for_next_rep) {
            s_training_down_frames = 0;
        } else if (bottom_condition) {
            if (s_training_down_frames == 0) {
                training_rep_eval_reset(&s_training_current_rep);
            }
            training_rep_eval_add_sample(&s_training_current_rep, m, valid_count);
            s_training_down_frames++;
        } else {
            s_training_down_frames = 0;
        }
        if (s_training_down_frames >= 2) {
            s_training_phase_active = true;
            s_training_phase_start_us = now_train_us;
            s_training_down_frames = 0;
            s_training_stand_frames = 0;
            training_set_cue_locked("深度已到，稳定起身。");
            strlcpy(s_training_detail, "下蹲阶段已确认，起身时保持膝盖稳定。", sizeof(s_training_detail));
        } else if (m.torso_lean > 48.0f) {
            training_set_cue_locked("躯干前倾偏大，胸口抬起一点。");
            strlcpy(s_training_detail, "躯干前倾偏大，优先抬胸并收紧核心。", sizeof(s_training_detail));
        } else if (m.knee_track_valid && m.knee_track_delta > 0.52f) {
            training_set_cue_locked("膝盖轨迹偏移，跟随脚尖方向。");
            strlcpy(s_training_detail, "膝盖和脚尖方向偏差较大，下一次下蹲放慢。", sizeof(s_training_detail));
        } else if (m.depth < 0.60f && (!m.knee_angle_valid || m.knee_angle > 140.0f) && m.hip_angle > 150.0f) {
            training_set_cue_locked("继续下蹲，髋部再低一点。");
            strlcpy(s_training_detail, "深度不足，髋部继续向后向下。", sizeof(s_training_detail));
        } else {
            training_set_cue_locked("动作接近标准，保持膝盖和脚尖同向。");
            strlcpy(s_training_detail, "节奏不错，保持膝盖与脚尖同向。", sizeof(s_training_detail));
        }
    } else {
        s_training_down_frames = 0;
        training_rep_eval_add_sample(&s_training_current_rep, m, valid_count);
        if (up_condition && rep_gap_ok) {
            s_training_up_frames++;
        } else {
            s_training_up_frames = 0;
        }
        if (s_training_up_frames >= 3 && bottom_duration_ok) {
            s_training_phase_active = false;
            s_training_up_frames = 0;
            s_training_stand_frames = 0;
            s_training_phase_start_us = 0;
            training_rep_eval_add_sample(&s_training_current_rep, m, valid_count);
            training_rep_eval_finalize(&s_training_current_rep);
            training_apply_rep_score_locked(s_training_current_rep);
            training_add_rep_locked(&s_training_current_rep);
            training_rep_eval_reset(&s_training_current_rep);
        } else if (m.torso_lean > 50.0f) {
            training_set_cue_locked("起身时背部再稳一点。");
            strlcpy(s_training_detail, "起身阶段背部不够稳定，放慢速度。", sizeof(s_training_detail));
        } else {
            training_set_cue_locked("深度已到，起身到站直后计数。");
            strlcpy(s_training_detail, "还未完全站直，继续起身后才会计数。", sizeof(s_training_detail));
        }
    }

    if (s_training_last_pose_us - s_training_last_debug_us >= 1000000) {
        s_training_last_debug_us = s_training_last_pose_us;
        ESP_LOGI(TAG,
                 "squat metrics: knee=%.0f%s hip=%.0f lean=%.0f depth=%.2f conf=%.2f bottom=%d up=%d phase=%d frames=%d/%d score=%d",
                 (double)m.knee_angle,
                 m.knee_angle_valid ? "" : "*",
                 (double)m.hip_angle,
                 (double)m.torso_lean,
                 (double)m.depth,
                 (double)m.confidence,
                 bottom_condition ? 1 : 0,
                 up_condition ? 1 : 0,
                 s_training_phase_active ? 1 : 0,
                 s_training_down_frames,
                 s_training_up_frames,
                 s_training_score);
    }
}

static void training_complete_locked(void)
{
    if (s_training_state == TrainingRunState::Running && s_training_start_us > 0) {
        s_training_elapsed_us += esp_timer_get_time() - s_training_start_us;
    }
    s_training_start_us = 0;
    s_training_state = TrainingRunState::Finished;
    s_training_phase_active = false;
    s_training_stand_frames = 0;
    s_training_down_frames = 0;
    s_training_up_frames = 0;
    s_training_rep_cooldown_frames = 0;
    s_training_last_rep_us = 0;
    s_training_phase_start_us = 0;
    s_training_needs_rearm = false;
    s_training_motion_ref = 0.0f;
    s_training_phase_peak_signal = 0.0f;
    s_training_hold_good_frames = 0;
    s_training_hold_bad_frames = 0;
    training_set_cue_locked("目标完成，训练结束。");
    training_record_save_current_locked("目标完成，训练结束。");
    music_send_cmd(MusicCmdType::VoicePrompt, (int)VoicePromptId::TrainingComplete);
}

static bool training_should_send_voice_locked(VoicePromptId prompt, int rep_score)
{
    const int64_t now_us = esp_timer_get_time();
    const bool same_prompt = prompt == s_training_last_voice_prompt;
    const bool recent_same = same_prompt && s_training_last_voice_us > 0 &&
                             now_us - s_training_last_voice_us < 12000000;
    (void)rep_score;
    if (recent_same) {
        return false;
    }
    s_training_last_voice_prompt = prompt;
    s_training_last_voice_us = now_us;
    return true;
}

static void training_add_rep_locked(const squat_rep_eval_t *rep_eval)
{
    const int64_t now_us = training_pose_now_us();
    if (s_training_last_rep_us > 0 && now_us - s_training_last_rep_us < training_min_rep_gap_us()) {
        ESP_LOGW(TAG,
                 "Training rep ignored by cooldown: action=%d gap=%lldus",
                 s_training_index,
                 (long long)(now_us - s_training_last_rep_us));
        s_training_phase_active = false;
        s_training_down_frames = 0;
        s_training_up_frames = 0;
        s_training_phase_peak_signal = 0.0f;
        s_training_needs_rearm = true;
        return;
    }

    const training_profile_t *profile = training_profile_by_index(s_training_index);
    const int target = training_target_count(profile);
    s_training_count = std::min(target, s_training_count + 1);
    s_training_last_rep_us = now_us;
    s_training_rep_cooldown_frames = 10;
    s_training_needs_rearm = true;
    s_training_phase_start_us = 0;
    const int reps_per_set = training_goal_reps_for_index(s_training_index);
    const int sets = training_goal_sets_for_index(s_training_index);
    if (profile->counted && reps_per_set > 0) {
        s_training_current_set = std::min<int>(sets, std::max(0, s_training_count - 1) / reps_per_set + 1);
    }
    if (target > 0 && s_training_count >= target) {
        training_complete_locked();
    } else {
        if (rep_eval) {
            char cue[120];
            snprintf(cue, sizeof(cue), "完成一次，本次%d分，继续保持节奏。", rep_eval->score);
            training_set_cue_locked(cue);
            if (rep_eval->good) {
                if (s_training_good_rep_streak > 0 && s_training_good_rep_streak % 5 == 0) {
                    music_send_cmd(MusicCmdType::VoicePrompt, (int)VoicePromptId::GoodStreak);
                }
            } else if (rep_eval->needs_voice && training_should_send_voice_locked(rep_eval->voice, rep_eval->score)) {
                music_send_cmd(MusicCmdType::VoicePrompt, (int)rep_eval->voice);
            }
        } else {
            training_set_cue_locked("完成一次，保持节奏继续。");
        }
    }
}

static bool training_should_send_voice_locked(VoicePromptId prompt, int rep_score);

static bool training_arm_metrics(const pose_result_t &pose,
                                 float min_score,
                                 float *elbow_angle,
                                 float *balance_delta,
                                 float *confidence)
{
    float angle_sum = 0.0f;
    float weight_sum = 0.0f;
    float left_angle = 0.0f;
    float right_angle = 0.0f;
    bool left_ok = false;
    bool right_ok = false;
    for (int side = 0; side < 2; ++side) {
        const bool left = side == 0;
        pose_keypoint_t shoulder = {};
        pose_keypoint_t elbow = {};
        pose_keypoint_t wrist = {};
        if (!training_get_keypoint(pose, left ? POSE_KPT_LEFT_SHOULDER : POSE_KPT_RIGHT_SHOULDER, min_score, &shoulder) ||
            !training_get_keypoint(pose, left ? POSE_KPT_LEFT_ELBOW : POSE_KPT_RIGHT_ELBOW, min_score, &elbow) ||
            !training_get_keypoint(pose, left ? POSE_KPT_LEFT_WRIST : POSE_KPT_RIGHT_WRIST, min_score, &wrist)) {
            continue;
        }
        const float angle = training_angle_deg(shoulder, elbow, wrist);
        const float w = std::max(0.05f, (shoulder.score + elbow.score + wrist.score) / 3.0f);
        angle_sum += angle * w;
        weight_sum += w;
        if (left) {
            left_angle = angle;
            left_ok = true;
        } else {
            right_angle = angle;
            right_ok = true;
        }
    }
    if (weight_sum <= 0.0f) {
        return false;
    }
    if (elbow_angle) {
        *elbow_angle = angle_sum / weight_sum;
    }
    if (balance_delta) {
        *balance_delta = (left_ok && right_ok) ? fabsf(left_angle - right_angle) : 0.0f;
    }
    if (confidence) {
        *confidence = training_clampf(weight_sum / 2.0f, 0.0f, 1.0f);
    }
    return true;
}

static bool training_body_line_angle(const pose_result_t &pose, float min_score, float *line_angle, float *torso_lean)
{
    pose_keypoint_t shoulder = {};
    pose_keypoint_t hip = {};
    pose_keypoint_t ankle = {};
    if (!training_avg_keypoint(pose, POSE_KPT_LEFT_SHOULDER, POSE_KPT_RIGHT_SHOULDER, min_score, &shoulder) ||
        !training_avg_keypoint(pose, POSE_KPT_LEFT_HIP, POSE_KPT_RIGHT_HIP, min_score, &hip) ||
        !training_avg_keypoint(pose, POSE_KPT_LEFT_ANKLE, POSE_KPT_RIGHT_ANKLE, min_score * 0.8f, &ankle)) {
        return false;
    }
    if (line_angle) {
        *line_angle = training_angle_deg(shoulder, hip, ankle);
    }
    if (torso_lean) {
        const float torso_dx = fabsf((float)shoulder.x - (float)hip.x);
        const float torso_dy = fabsf((float)shoulder.y - (float)hip.y);
        *torso_lean = atan2f(torso_dx, std::max(1.0f, torso_dy)) * 57.2957795f;
    }
    return true;
}

static void training_update_press_locked(const pose_result_t &pose, int valid_count, float avg_score, bool bench)
{
    training_tick_rep_cooldown_locked();
    float elbow = 180.0f;
    float balance = 0.0f;
    float conf = 0.0f;
    if (!training_arm_metrics(pose, 0.14f, &elbow, &balance, &conf)) {
        training_set_cue_locked(bench ? "请让肩、肘、腕进入画面。" : "请让肩、肘、腕进入画面。");
        training_set_live_scores_locked(0, 0, 0, 0, 0, 0, training_track_score(valid_count, avg_score), "上肢关键点不足。");
        s_training_down_frames = 0;
        s_training_up_frames = 0;
        return;
    }

    float body_line = 175.0f;
    float torso_lean = 0.0f;
    const bool body_ok = training_body_line_angle(pose, 0.12f, &body_line, &torso_lean);
    const bool orientation_ok = body_ok && body_line >= 142.0f && torso_lean >= (bench ? 32.0f : 44.0f);
    if (!orientation_ok) {
        training_set_cue_locked(bench ? "卧推请从侧前方拍摄，让肩髋和手臂都入镜。" :
                                        "俯卧撑请侧面或45度侧前方拍摄，让肩髋脚踝横向入镜。");
        training_set_live_scores_locked(0,
                                        0,
                                        0,
                                        body_ok ? training_range_score(body_line, 158.0f, 180.0f, 34.0f) : 0,
                                        body_ok ? training_score_from01(torso_lean / 70.0f) : 0,
                                        0,
                                        training_track_score(valid_count, avg_score),
                                        bench ? "卧推需要侧前方视角，避免站姿或手臂摆动误计数。" :
                                                "俯卧撑需要身体横向入镜，否则不进入计数。");
        if (training_should_send_voice_locked(bench ? VoicePromptId::BenchWrist : VoicePromptId::PushupHip, 0)) {
            music_send_cmd(MusicCmdType::VoicePrompt, (int)(bench ? VoicePromptId::BenchWrist : VoicePromptId::PushupHip));
        }
        s_training_phase_active = false;
        s_training_down_frames = 0;
        s_training_up_frames = 0;
        s_training_phase_peak_signal = 0.0f;
        s_training_needs_rearm = true;
        return;
    }
    const bool low_condition = elbow <= (bench ? 112.0f : 108.0f);
    const bool high_condition = elbow >= 154.0f;
    const int track = training_track_score(valid_count, avg_score);
    const int range = training_range_score(elbow, 72.0f, bench ? 112.0f : 108.0f, 54.0f);
    const int line_score = bench ? 82 : (body_ok ? training_range_score(body_line, 158.0f, 180.0f, 34.0f) : 62);
    const int sync_score = balance > 0.0f ? training_score_from01(1.0f - (balance - 12.0f) / 42.0f) : 78;
    const int total = std::max(0, std::min(100, (int)lroundf(range * 0.34f + line_score * 0.24f + sync_score * 0.20f + track * 0.22f)));
    training_set_live_scores_locked(total, range, sync_score, line_score, line_score, sync_score, track,
                                    bench ? "卧推实时评分：幅度、左右同步和稳定性。" :
                                            "俯卧撑实时评分：下降幅度、身体线条和左右同步。");
    s_training_knee_angle = elbow;
    s_training_hip_angle = body_line;
    s_training_torso_lean = torso_lean;
    s_training_depth_ratio = training_clampf((180.0f - elbow) / 110.0f, 0.0f, 1.0f);
    s_training_symmetry_delta = balance;

    if (!s_training_phase_active) {
        training_clear_rearm_when_ready_locked(high_condition, 5);
        const bool ready_for_rep = training_can_start_rep_locked(4);
        if (low_condition && ready_for_rep) {
            s_training_down_frames++;
        } else if (!low_condition) {
            s_training_down_frames = 0;
        }
        if (s_training_down_frames >= 3) {
            s_training_phase_active = true;
            s_training_phase_start_us = training_pose_now_us();
            s_training_up_frames = 0;
            s_training_down_frames = 0;
            training_simple_rep_reset_locked(range, sync_score, line_score, line_score, sync_score, track);
            training_set_cue_locked(bench ? "下放到位，稳定推起。" : "下降到位，推起时保持身体一条线。");
        } else if (!bench && body_ok && body_line < 150.0f) {
            training_set_cue_locked("核心收紧，肩髋脚跟保持一条线。");
            if (training_should_send_voice_locked(VoicePromptId::PushupHip, total)) {
                music_send_cmd(MusicCmdType::VoicePrompt, (int)VoicePromptId::PushupHip);
            }
        } else if (elbow > (bench ? 128.0f : 124.0f)) {
            training_set_cue_locked(bench ? "继续下放，肘部角度还不够。" : "继续下降，胸口再靠近地面。");
        } else {
            training_set_cue_locked(bench ? "继续下放，保持左右同步。" : "继续下降，胸口再低一点。");
        }
        return;
    }

    const int64_t now_us = training_pose_now_us();
    const bool duration_ok = s_training_phase_start_us > 0 &&
                             now_us - s_training_phase_start_us >= (bench ? 650000 : 750000);
    training_simple_rep_sample_locked(range, sync_score, line_score, line_score, sync_score, track);
    if (high_condition) {
        s_training_up_frames++;
    } else {
        s_training_up_frames = 0;
    }
    if (s_training_up_frames >= 4 && duration_ok) {
        s_training_phase_active = false;
        s_training_up_frames = 0;
        s_training_stand_frames = 0;
        const int rep_range = s_training_simple_rep_valid ? s_training_simple_rep_depth : range;
        const int rep_sync = s_training_simple_rep_valid ? s_training_simple_rep_knee : sync_score;
        const int rep_line = s_training_simple_rep_valid ? s_training_simple_rep_hip : line_score;
        VoicePromptId voice = bench ? VoicePromptId::BenchRange : VoicePromptId::PushupDepth;
        bool needs_voice = rep_range < 58 || rep_sync < 55 || (!bench && rep_line < 65);
        const char *detail = bench ? "本次卧推完成，重点看推起幅度和左右同步。" :
                                     "本次俯卧撑完成，重点看下降幅度和身体线条。";
        if (!bench && rep_line < 65) {
            voice = rep_line < 48 ? VoicePromptId::PushupHip : VoicePromptId::PushupBodyLine;
            needs_voice = true;
            detail = "本次俯卧撑身体线条不够稳定，下一次收紧核心。";
        } else if (!bench && rep_range < 58) {
            voice = VoicePromptId::PushupElbow;
            needs_voice = true;
            detail = "本次俯卧撑下降幅度不够，下一次胸口再靠近地面。";
        } else if (bench && rep_sync < 55) {
            voice = rep_sync < 42 ? VoicePromptId::BenchWrist : VoicePromptId::BenchShoulder;
            needs_voice = true;
            detail = "本次卧推左右不同步，下一次两侧同时推起。";
        }
        training_commit_simple_rep_best_locked(range, sync_score, line_score, line_score, sync_score, track, voice, needs_voice, detail);
    } else {
        training_set_cue_locked(bench ? "继续推起到手臂伸展。" : "继续推起到手臂伸展。");
    }
}

static void training_update_pullup_locked(const pose_result_t &pose, int valid_count, float avg_score)
{
    training_tick_rep_cooldown_locked();
    float elbow = 180.0f;
    float balance = 0.0f;
    float conf = 0.0f;
    pose_keypoint_t shoulder = {};
    pose_keypoint_t wrist = {};
    pose_keypoint_t nose = {};
    pose_keypoint_t hip = {};
    if (!training_arm_metrics(pose, 0.14f, &elbow, &balance, &conf) ||
        !training_avg_keypoint(pose, POSE_KPT_LEFT_SHOULDER, POSE_KPT_RIGHT_SHOULDER, 0.14f, &shoulder) ||
        !training_avg_keypoint(pose, POSE_KPT_LEFT_WRIST, POSE_KPT_RIGHT_WRIST, 0.12f, &wrist) ||
        !training_get_keypoint(pose, 0, 0.12f, &nose) ||
        !training_avg_keypoint(pose, POSE_KPT_LEFT_HIP, POSE_KPT_RIGHT_HIP, 0.12f, &hip)) {
        training_set_cue_locked("请让头部、肩、肘、腕进入画面。");
        training_set_live_scores_locked(0, 0, 0, 0, 0, 0, training_track_score(valid_count, avg_score), "引体向上关键点不足。");
        s_training_phase_active = false;
        s_training_down_frames = 0;
        s_training_up_frames = 0;
        s_training_phase_peak_signal = 0.0f;
        s_training_needs_rearm = true;
        return;
    }

    const float height_gap = (float)nose.y - (float)wrist.y;
    const float shoulder_to_wrist = std::max(48.0f, fabsf((float)wrist.y - (float)shoulder.y));
    const float body_sway = fabsf((float)hip.x - (float)shoulder.x) / std::max(80.0f, fabsf((float)hip.y - (float)shoulder.y));
    const bool overhead_ok = wrist.y < shoulder.y + 52;
    if (!overhead_ok) {
        training_set_cue_locked("引体向上请让双手在肩上方入镜，正面或侧面都可以。");
        training_set_live_scores_locked(0, 0, 0, 0, 0, 0, training_track_score(valid_count, avg_score), "未识别到上方握杆姿态，不进入计数。");
        if (training_should_send_voice_locked(VoicePromptId::PullupGrip, 0)) {
            music_send_cmd(MusicCmdType::VoicePrompt, (int)VoicePromptId::PullupGrip);
        }
        s_training_phase_active = false;
        s_training_down_frames = 0;
        s_training_up_frames = 0;
        s_training_needs_rearm = true;
        return;
    }
    const bool bottom_condition = overhead_ok && elbow >= 138.0f && height_gap >= shoulder_to_wrist * 0.36f;
    if (!s_training_phase_active && bottom_condition) {
        if (s_training_motion_ref <= 1.0f) {
            s_training_motion_ref = height_gap;
        } else {
            s_training_motion_ref = std::max(s_training_motion_ref * 0.88f + height_gap * 0.12f, height_gap);
        }
    }
    const float bottom_ref = s_training_motion_ref > 1.0f ? s_training_motion_ref : std::max(height_gap, shoulder_to_wrist * 0.82f);
    const float travel = std::max(0.0f, bottom_ref - height_gap);
    const float required_travel = std::max(36.0f, shoulder_to_wrist * 0.28f);
    const bool top_by_travel = travel >= required_travel && height_gap <= bottom_ref - required_travel * 0.74f;
    const bool top_condition = overhead_ok &&
                               elbow <= 136.0f &&
                               top_by_travel;
    const int height_score = training_score_from01((travel - required_travel * 0.72f) / std::max(42.0f, required_travel * 1.18f));
    const int arm_score = training_range_score(elbow, 72.0f, 132.0f, 58.0f);
    const int sway_score = training_score_from01(1.0f - (body_sway - 0.20f) / 0.42f);
    const int sync_score = balance > 0.0f ? training_score_from01(1.0f - (balance - 14.0f) / 44.0f) : 78;
    const int track = training_track_score(valid_count, avg_score);
    const int total = std::max(0, std::min(100, (int)lroundf(height_score * 0.34f + arm_score * 0.22f + sway_score * 0.20f + sync_score * 0.10f + track * 0.14f)));
    training_set_live_scores_locked(total, height_score, sync_score, arm_score, sway_score, sync_score, track, "引体向上实时评分：高度、摆动和左右同步。");
    s_training_knee_angle = elbow;
    s_training_hip_angle = 0.0f;
    s_training_torso_lean = body_sway * 57.0f;
    s_training_depth_ratio = training_clampf(1.0f - height_gap / 220.0f, 0.0f, 1.0f);
    s_training_symmetry_delta = balance;

    if (!s_training_phase_active) {
        training_clear_rearm_when_ready_locked(bottom_condition, 4);
        if (top_condition && training_can_start_rep_locked(4)) {
            s_training_down_frames++;
        } else if (!top_condition) {
            s_training_down_frames = 0;
        }
        if (s_training_down_frames >= 2) {
            s_training_phase_active = true;
            s_training_phase_start_us = training_pose_now_us();
            s_training_phase_peak_signal = travel;
            s_training_up_frames = 0;
            s_training_down_frames = 0;
            training_simple_rep_reset_locked(height_score, sync_score, arm_score, sway_score, sync_score, track);
            training_set_cue_locked("拉到位，控制身体回到底部。");
        } else if (sway_score < 55) {
            training_set_cue_locked("身体摆动偏大，先稳住再拉。");
        } else if (s_training_stand_frames < 3) {
            training_set_cue_locked("先到底部伸展，再开始一次完整引体。");
        } else if (elbow > 132.0f) {
            training_set_cue_locked("继续屈肘向上拉，下巴尽量靠近横杠。");
        } else {
            training_set_cue_locked("继续向上拉，胸口靠近横杠。");
        }
        return;
    }
    const int64_t now_us = training_pose_now_us();
    s_training_phase_peak_signal = std::max(s_training_phase_peak_signal, travel);
    const bool duration_ok = s_training_phase_start_us > 0 && now_us - s_training_phase_start_us >= 750000;
    const bool had_real_pull = s_training_phase_peak_signal >= required_travel;
    const bool returned_bottom = had_real_pull &&
                                 bottom_condition &&
                                 height_gap >= bottom_ref - std::max(40.0f, required_travel * 0.55f) &&
                                 elbow >= 136.0f;
    training_simple_rep_sample_locked(height_score, sync_score, arm_score, sway_score, sync_score, track);
    if (returned_bottom) {
        s_training_up_frames++;
    } else {
        s_training_up_frames = 0;
    }
    if (s_training_up_frames >= 2 && duration_ok) {
        s_training_phase_active = false;
        s_training_stand_frames = 0;
        s_training_up_frames = 0;
        s_training_phase_peak_signal = 0.0f;
        const int rep_height = s_training_simple_rep_valid ? s_training_simple_rep_depth : height_score;
        const int rep_arm = s_training_simple_rep_valid ? s_training_simple_rep_hip : arm_score;
        const int rep_sway = s_training_simple_rep_valid ? s_training_simple_rep_torso : sway_score;
        VoicePromptId voice = rep_height < 54 ? VoicePromptId::PullupHeight :
                              (rep_sway < 46 ? VoicePromptId::PullupSwing : VoicePromptId::PullupControl);
        bool needs_voice = rep_height < 54 || rep_sway < 46 || rep_arm < 46;
        const char *detail = rep_height < 54 ? "本次引体向上高度不够，下一次胸口主动靠近横杠。" :
                             (rep_sway < 46 ? "本次引体向上身体摆动偏大，下一次先稳住核心。" :
                              (rep_arm < 46 ? "本次引体向上手臂行程不够完整，下降到底部再发力。" :
                                                "本次引体向上完成，节奏保持稳定。"));
        training_commit_simple_rep_best_locked(height_score, sync_score, arm_score, sway_score, sync_score, track, voice, needs_voice, detail);
    } else {
        if (duration_ok && s_training_up_frames == 0 && training_should_send_voice_locked(VoicePromptId::PullupControl, total)) {
            music_send_cmd(MusicCmdType::VoicePrompt, (int)VoicePromptId::PullupControl);
        }
        training_set_cue_locked("控制下降到底部伸展后计数。");
    }
}

static void training_update_deadlift_locked(const pose_result_t &pose, int valid_count, float avg_score)
{
    training_tick_rep_cooldown_locked();
    squat_metrics_t m = {};
    if (!training_squat_metrics(pose, &m)) {
        training_set_cue_locked("请让肩、髋、膝、踝进入画面。");
        training_set_live_scores_locked(0, 0, 0, 0, 0, 0, training_track_score(valid_count, avg_score), "硬拉关键点不足。");
        s_training_down_frames = 0;
        s_training_up_frames = 0;
        return;
    }
    const bool bottom_condition = m.hip_angle <= 138.0f && m.depth >= 0.42f;
    const bool lockout_condition = m.hip_angle >= 158.0f && (!m.knee_angle_valid || m.knee_angle >= 145.0f) && m.torso_lean <= 38.0f;
    const int hip_score = training_range_score(m.hip_angle, 88.0f, 138.0f, 58.0f);
    const int back_score = training_score_from01(1.0f - (m.torso_lean - 34.0f) / 34.0f);
    const int knee_score = m.knee_angle_valid ? training_range_score(m.knee_angle, 112.0f, 170.0f, 46.0f) : 72;
    const int balance = m.both_sides ? training_score_from01(1.0f - (m.symmetry_delta - 16.0f) / 38.0f) : 78;
    const int track = training_track_score(valid_count, avg_score);
    const int total = std::max(0, std::min(100, (int)lroundf(hip_score * 0.28f + back_score * 0.28f + knee_score * 0.16f + balance * 0.10f + track * 0.18f)));
    training_set_live_scores_locked(total, hip_score, knee_score, hip_score, back_score, balance, track, "硬拉实时评分：髋部发力、背部稳定和锁定。");
    s_training_knee_angle = m.knee_angle;
    s_training_hip_angle = m.hip_angle;
    s_training_torso_lean = m.torso_lean;
    s_training_depth_ratio = m.depth;
    s_training_symmetry_delta = m.symmetry_delta;

    if (!s_training_phase_active) {
        training_clear_rearm_when_ready_locked(lockout_condition, 5);
        if (bottom_condition && training_can_start_rep_locked(4)) {
            s_training_down_frames++;
        } else if (!bottom_condition) {
            s_training_down_frames = 0;
        }
        if (s_training_down_frames >= 3) {
            s_training_phase_active = true;
            s_training_phase_start_us = training_pose_now_us();
            s_training_up_frames = 0;
            training_simple_rep_reset_locked(hip_score, knee_score, hip_score, back_score, balance, track);
            training_set_cue_locked("底部到位，髋部发力站直。");
        } else if (back_score < 55) {
            training_set_cue_locked("背部再稳一点，抬胸收紧核心。");
        } else if (knee_score < 55) {
            training_set_cue_locked("膝髋节奏不稳，先髋部向后再屈膝。");
        } else {
            training_set_cue_locked("髋部向后，保持背部稳定。");
        }
        return;
    }
    const int64_t now_us = training_pose_now_us();
    const bool duration_ok = s_training_phase_start_us > 0 && now_us - s_training_phase_start_us >= 900000;
    training_simple_rep_sample_locked(hip_score, knee_score, hip_score, back_score, balance, track);
    if (lockout_condition) {
        s_training_up_frames++;
    } else {
        s_training_up_frames = 0;
    }
    if (s_training_up_frames >= 4 && duration_ok) {
        s_training_phase_active = false;
        s_training_stand_frames = 0;
        s_training_up_frames = 0;
        const int rep_hip = s_training_simple_rep_valid ? s_training_simple_rep_depth : hip_score;
        const int rep_knee = s_training_simple_rep_valid ? s_training_simple_rep_knee : knee_score;
        const int rep_back = s_training_simple_rep_valid ? s_training_simple_rep_torso : back_score;
        VoicePromptId voice = rep_back < 52 ? VoicePromptId::DeadliftBack :
                              (rep_knee < 52 ? VoicePromptId::DeadliftKnee : VoicePromptId::DeadliftHip);
        bool needs_voice = rep_back < 52 || rep_hip < 52 || rep_knee < 52;
        const char *detail = rep_back < 52 ? "本次硬拉背部不够稳定，下一次抬胸收紧核心。" :
                             (rep_knee < 52 ? "本次硬拉膝髋节奏不协调，下一次先髋后膝。" :
                              (rep_hip < 52 ? "本次硬拉髋部发力不足，下一次先髋后膝。" :
                                               "本次硬拉完成，锁定阶段保持稳定。"));
        training_commit_simple_rep_best_locked(hip_score, knee_score, hip_score, back_score, balance, track, voice, needs_voice, detail);
    } else {
        training_set_cue_locked("继续站直到髋膝锁定。");
    }
}

static void training_update_plank_locked(const pose_result_t &pose, int valid_count, float avg_score)
{
    training_tick_rep_cooldown_locked();
    float line_angle = 0.0f;
    float torso_lean = 0.0f;
    const int64_t now_us = training_pose_now_us();
    if (!training_body_line_angle(pose, 0.12f, &line_angle, &torso_lean)) {
        training_set_cue_locked("请让肩、髋、踝进入画面。");
        training_set_live_scores_locked(0, 0, 0, 0, 0, 0, training_track_score(valid_count, avg_score), "平板支撑关键点不足。");
        s_training_hold_good_frames = 0;
        if (++s_training_hold_bad_frames >= 5 && s_training_start_us > 0) {
            s_training_elapsed_us += now_us - s_training_start_us;
            s_training_start_us = 0;
            training_set_cue_locked("支撑中断，重新进入平板姿态后再开始计时。");
            if (training_should_send_voice_locked(VoicePromptId::PlankRestart, 0)) {
                music_send_cmd(MusicCmdType::VoicePrompt, (int)VoicePromptId::PlankRestart);
            }
        }
        return;
    }
    const bool plank_shape = torso_lean >= 36.0f && line_angle >= 128.0f && valid_count >= 7;
    const bool stable_hold = torso_lean >= 42.0f && line_angle >= 142.0f && valid_count >= 8;
    const bool hard_break = torso_lean < 24.0f || line_angle < 108.0f || valid_count < 5;
    const int line_score = training_range_score(line_angle, 154.0f, 180.0f, 40.0f);
    const int torso_score = training_score_from01((torso_lean - 30.0f) / 42.0f);
    const int track = training_track_score(valid_count, avg_score);
    const int total = std::max(0, std::min(100, (int)lroundf(line_score * 0.46f + torso_score * 0.24f + track * 0.30f)));
    training_set_live_scores_locked(total, line_score, 80, line_score, torso_score, 80, track, "平板支撑实时评分：肩髋踝连线和核心稳定。");
    s_training_knee_angle = 0.0f;
    s_training_hip_angle = line_angle;
    s_training_torso_lean = torso_lean;
    s_training_depth_ratio = line_angle / 180.0f;
    s_training_symmetry_delta = 0.0f;

    if (stable_hold) {
        s_training_hold_good_frames = std::min(s_training_hold_good_frames + 1, 12);
        s_training_hold_bad_frames = 0;
        if (s_training_start_us == 0 && s_training_hold_good_frames >= 3) {
            s_training_start_us = now_us;
            training_set_cue_locked("已进入平板支撑，开始计时。");
            if (training_should_send_voice_locked(VoicePromptId::PlankHold, total)) {
                music_send_cmd(MusicCmdType::VoicePrompt, (int)VoicePromptId::PlankHold);
            }
        } else if (line_score < 68) {
            training_set_cue_locked("髋部位置再调整，肩髋脚跟尽量一条线。");
        } else {
            training_set_cue_locked("保持核心收紧，呼吸稳定。");
        }
        return;
    }

    s_training_hold_good_frames = 0;
    if (plank_shape && s_training_start_us > 0) {
        s_training_hold_bad_frames = 0;
        if (line_angle < 142.0f) {
            training_set_cue_locked("腰部有下塌趋势，收紧核心继续保持。");
            if (training_should_send_voice_locked(VoicePromptId::PlankSag, total)) {
                music_send_cmd(MusicCmdType::VoicePrompt, (int)VoicePromptId::PlankSag);
            }
        } else if (line_angle > 178.0f || torso_lean < 42.0f) {
            training_set_cue_locked("髋部略高，放平一点但继续计时。");
            if (training_should_send_voice_locked(VoicePromptId::PlankHip, total)) {
                music_send_cmd(MusicCmdType::VoicePrompt, (int)VoicePromptId::PlankHip);
            }
        } else {
            training_set_cue_locked("动作接近标准，继续保持。");
        }
        return;
    }

    if (hard_break || !plank_shape) {
        s_training_hold_bad_frames++;
        if (s_training_start_us > 0 && s_training_hold_bad_frames >= 4) {
            s_training_elapsed_us += now_us - s_training_start_us;
            s_training_start_us = 0;
            training_set_cue_locked("支撑中断，重新进入平板姿态后再开始计时。");
            if (training_should_send_voice_locked(VoicePromptId::PlankRestart, total)) {
                music_send_cmd(MusicCmdType::VoicePrompt, (int)VoicePromptId::PlankRestart);
            }
        } else if (s_training_start_us == 0) {
            training_set_cue_locked("趴下并撑起身体，肩髋脚跟成一条线后开始计时。");
        }
    } else {
        training_set_cue_locked("支撑姿态接近标准，再稳定一下开始计时。");
    }
}

static void training_update_curl_locked(const pose_result_t &pose, int valid_count, float avg_score)
{
    training_tick_rep_cooldown_locked();
    float elbow = 180.0f;
    float balance = 0.0f;
    float conf = 0.0f;
    if (!training_arm_metrics(pose, 0.10f, &elbow, &balance, &conf)) {
        training_set_cue_locked("请让肩、肘、腕进入画面。");
        training_set_live_scores_locked(0, 0, 0, 0, 0, 0, training_track_score(valid_count, avg_score), "哑铃弯举关键点不足。");
        s_training_down_frames = 0;
        s_training_up_frames = 0;
        return;
    }

    const bool bottom_condition = elbow >= 128.0f;
    const bool top_condition = elbow <= 108.0f;
    const int range_score = training_range_score(elbow, 58.0f, 112.0f, 62.0f);
    const int sync_score = balance > 0.0f ? training_score_from01(1.0f - (balance - 22.0f) / 54.0f) : 80;
    const int track = training_track_score(valid_count, avg_score);
    const int total = std::max(0, std::min(100, (int)lroundf(range_score * 0.46f + sync_score * 0.22f + track * 0.32f)));
    training_set_live_scores_locked(total, range_score, sync_score, range_score, 82, sync_score, track,
                                    "哑铃弯举实时评分：屈肘幅度、左右同步和识别稳定。");
    s_training_knee_angle = elbow;
    s_training_hip_angle = 0.0f;
    s_training_torso_lean = 0.0f;
    s_training_depth_ratio = training_clampf((170.0f - elbow) / 120.0f, 0.0f, 1.0f);
    s_training_symmetry_delta = balance;

    if (!s_training_phase_active) {
        training_clear_rearm_when_ready_locked(bottom_condition, 3);
        if (top_condition && training_can_start_rep_locked(3)) {
            s_training_down_frames++;
        } else if (!top_condition) {
            s_training_down_frames = 0;
        }
        if (s_training_down_frames >= 2) {
            s_training_phase_active = true;
            s_training_phase_start_us = training_pose_now_us();
            s_training_up_frames = 0;
            s_training_down_frames = 0;
            training_simple_rep_reset_locked(range_score, sync_score, range_score, 82, sync_score, track);
            training_set_cue_locked("弯举到位，控制下放到底部。");
        } else if (s_training_stand_frames < 3) {
            training_set_cue_locked("先把手臂放到底部伸展，再开始一次弯举。");
        } else if (elbow > 116.0f) {
            training_set_cue_locked("继续屈肘向上，动作幅度还不够。");
        } else {
            training_set_cue_locked("上臂保持稳定，避免身体借力。");
        }
        return;
    }

    const int64_t now_us = training_pose_now_us();
    const bool duration_ok = s_training_phase_start_us > 0 && now_us - s_training_phase_start_us >= 800000;
    training_simple_rep_sample_locked(range_score, sync_score, range_score, 82, sync_score, track);
    if (bottom_condition) {
        s_training_up_frames++;
    } else {
        s_training_up_frames = 0;
    }
    if (s_training_up_frames >= 3 && duration_ok) {
        s_training_phase_active = false;
        s_training_up_frames = 0;
        s_training_stand_frames = 0;
        const int rep_range = s_training_simple_rep_valid ? s_training_simple_rep_depth : range_score;
        const int rep_sync = s_training_simple_rep_valid ? s_training_simple_rep_knee : sync_score;
        const bool needs_voice = rep_range < 54 || rep_sync < 50;
        const VoicePromptId voice = rep_range < 54 ? VoicePromptId::CurlRange : VoicePromptId::CurlSwing;
        const char *detail = rep_range < 54 ?
                             "本次哑铃弯举幅度不够，下次继续屈肘到顶点再控制下放。" :
                             (rep_sync < 50 ? "本次弯举左右不同步或有借力，下次放慢并稳定上臂。" :
                                                "本次哑铃弯举完成，节奏保持稳定。");
        training_commit_simple_rep_best_locked(range_score, sync_score, range_score, 82, sync_score, track, voice, needs_voice, detail);
    } else {
        training_set_cue_locked("控制下放到底部伸展后计数。");
    }
}

static void training_update_situp_locked(const pose_result_t &pose, int valid_count, float avg_score)
{
    training_tick_rep_cooldown_locked();
    pose_keypoint_t shoulder = {};
    pose_keypoint_t hip = {};
    pose_keypoint_t knee = {};
    pose_keypoint_t nose = {};
    if (!training_avg_keypoint(pose, POSE_KPT_LEFT_SHOULDER, POSE_KPT_RIGHT_SHOULDER, 0.10f, &shoulder) ||
        !training_avg_keypoint(pose, POSE_KPT_LEFT_HIP, POSE_KPT_RIGHT_HIP, 0.10f, &hip) ||
        !training_avg_keypoint(pose, POSE_KPT_LEFT_KNEE, POSE_KPT_RIGHT_KNEE, 0.08f, &knee)) {
        training_set_cue_locked("请让肩、髋、膝进入画面，建议侧面拍摄。");
        training_set_live_scores_locked(0, 0, 0, 0, 0, 0, training_track_score(valid_count, avg_score), "仰卧起坐关键点不足。");
        s_training_down_frames = 0;
        s_training_up_frames = 0;
        return;
    }

    const bool has_nose = training_get_keypoint(pose, 0, 0.08f, &nose);
    const float trunk_angle = training_angle_deg(shoulder, hip, knee);
    const bool bottom_condition = trunk_angle >= 132.0f;
    const bool top_condition = trunk_angle <= 126.0f;
    const int range_score = training_range_score(trunk_angle, 82.0f, 126.0f, 72.0f);
    const int neck_score = has_nose ? training_score_from01(1.0f - (fabsf((float)nose.x - (float)shoulder.x) - 135.0f) / 210.0f) : 78;
    const int track = training_track_score(valid_count, avg_score);
    const int total = std::max(0, std::min(100, (int)lroundf(range_score * 0.44f + neck_score * 0.20f + track * 0.36f)));
    training_set_live_scores_locked(total, range_score, neck_score, range_score, neck_score, 80, track,
                                    "仰卧起坐实时评分：卷腹幅度、颈部稳定和识别稳定。");
    s_training_knee_angle = 0.0f;
    s_training_hip_angle = trunk_angle;
    s_training_torso_lean = 0.0f;
    s_training_depth_ratio = training_clampf((160.0f - trunk_angle) / 88.0f, 0.0f, 1.0f);
    s_training_symmetry_delta = 0.0f;

    if (!s_training_phase_active) {
        training_clear_rearm_when_ready_locked(bottom_condition, 3);
        if (top_condition && training_can_start_rep_locked(3)) {
            s_training_down_frames++;
        } else if (!top_condition) {
            s_training_down_frames = 0;
        }
        if (s_training_down_frames >= 2) {
            s_training_phase_active = true;
            s_training_phase_start_us = training_pose_now_us();
            s_training_up_frames = 0;
            s_training_down_frames = 0;
            training_simple_rep_reset_locked(range_score, neck_score, range_score, neck_score, 80, track);
            training_set_cue_locked("卷起到位，控制身体回到底部。");
        } else if (s_training_stand_frames < 3) {
            training_set_cue_locked("先回到底部躺平位置，再开始一次仰卧起坐。");
        } else if (!top_condition) {
            training_set_cue_locked("继续腹部发力卷起，幅度再完整一点。");
        }
        return;
    }

    const int64_t now_us = training_pose_now_us();
    const bool duration_ok = s_training_phase_start_us > 0 && now_us - s_training_phase_start_us >= 900000;
    training_simple_rep_sample_locked(range_score, neck_score, range_score, neck_score, 80, track);
    if (bottom_condition) {
        s_training_up_frames++;
    } else {
        s_training_up_frames = 0;
    }
    if (s_training_up_frames >= 3 && duration_ok) {
        s_training_phase_active = false;
        s_training_up_frames = 0;
        s_training_stand_frames = 0;
        const int rep_range = s_training_simple_rep_valid ? s_training_simple_rep_depth : range_score;
        const int rep_neck = s_training_simple_rep_valid ? s_training_simple_rep_knee : neck_score;
        const bool needs_voice = rep_range < 54 || rep_neck < 48;
        const VoicePromptId voice = rep_range < 54 ? VoicePromptId::SitupRange : VoicePromptId::SitupNeck;
        const char *detail = rep_range < 54 ?
                             "本次仰卧起坐幅度不够，下次用腹部卷起到更高位置。" :
                             (rep_neck < 48 ? "本次仰卧起坐颈部不够稳定，下次不要用手猛拉头部。" :
                                                "本次仰卧起坐完成，继续控制下放节奏。");
        training_commit_simple_rep_best_locked(range_score, neck_score, range_score, neck_score, 80, track, voice, needs_voice, detail);
    } else {
        training_set_cue_locked("控制下放回到底部后计数。");
    }
}

static void training_update_from_pc_pose(const pose_result_t &pose, int valid_count, float pc_fps)
{
    training_lock();
    if (s_training_state != TrainingRunState::Running) {
        s_training_valid_kpts = valid_count;
        s_training_pc_fps = pc_fps;
        training_unlock();
        return;
    }

    const training_profile_t *profile = training_profile_by_index(s_training_index);
    s_training_valid_kpts = valid_count;
    s_training_pc_fps = pc_fps;
    s_training_last_pose_us = esp_timer_get_time();

    float score_sum = 0.0f;
    int score_n = 0;
    for (const pose_keypoint_t &kp : pose.keypoints) {
        if (kp.score >= 0.18f) {
            score_sum += kp.score;
            score_n++;
        }
    }
    const float avg_score = score_n > 0 ? score_sum / (float)score_n : 0.0f;
    if (s_training_index != 0 && s_training_index != 5) {
        s_training_score = std::max(0, std::min(100, (int)lroundf(avg_score * 70.0f + (float)valid_count * 2.0f)));
    }

    const uint32_t elapsed_ms = training_elapsed_ms_locked(s_training_last_pose_us);
    const uint32_t target_duration_sec = training_goal_duration_for_index(s_training_index);
    if (!profile->counted && target_duration_sec > 0) {
        if (s_training_index == 5) {
            training_update_plank_locked(pose, valid_count, avg_score);
            const uint32_t plank_elapsed_ms = training_elapsed_ms_locked(s_training_last_pose_us);
            if (plank_elapsed_ms >= target_duration_sec * 1000U) {
                training_complete_locked();
            }
            training_unlock();
            return;
        }
        if (elapsed_ms >= target_duration_sec * 1000U) {
            training_complete_locked();
        } else if (s_training_index != 5) {
            training_set_cue_locked("保持稳定，注意呼吸。");
        }
        training_unlock();
        return;
    }

    if (s_training_index == 0) {
        training_update_squat_locked(pose, valid_count, avg_score);
        training_unlock();
        return;
    }

    if (s_training_index == 1) {
        training_update_press_locked(pose, valid_count, avg_score, false);
    } else if (s_training_index == 2) {
        training_update_press_locked(pose, valid_count, avg_score, true);
    } else if (s_training_index == 3) {
        training_update_pullup_locked(pose, valid_count, avg_score);
    } else if (s_training_index == 4) {
        training_update_deadlift_locked(pose, valid_count, avg_score);
    } else if (s_training_index == 6) {
        training_update_curl_locked(pose, valid_count, avg_score);
    } else if (s_training_index == 7) {
        training_update_situp_locked(pose, valid_count, avg_score);
    } else {
        training_set_cue_locked(profile->counted ? "PC骨架已跟踪，动作计数规则待完善。" : "保持稳定，注意呼吸。");
    }

    training_unlock();
}

static void update_correction_ui_from_snapshot(const training_snapshot_t &snap)
{
    char text[180];
    const bool has_training = snap.profile != nullptr;

    if (lbl_corr_summary) {
        snprintf(text,
                 sizeof(text),
                 "%s | 总分 %d | PC %.1f FPS",
                 snap.profile ? snap.profile->name : "--",
                 snap.score,
                 (double)snap.pc_fps);
        lv_label_set_text(lbl_corr_summary, text);
    }
    if (lbl_corr_depth) {
        snprintf(text,
                 sizeof(text),
                 "幅度 %d | 比例 %.2f | 动作幅度越完整越好",
                 has_training ? snap.score_depth : 0,
                 (double)snap.depth_ratio);
        lv_label_set_text(lbl_corr_depth, text);
    }
    if (lbl_corr_knee) {
        snprintf(text,
                 sizeof(text),
                 "膝盖轨迹 %d | 膝角 %.0f° | 膝盖跟随脚尖",
                 has_training ? snap.score_knee : 0,
                 (double)snap.knee_angle);
        lv_label_set_text(lbl_corr_knee, text);
    }
    if (lbl_corr_hip) {
        snprintf(text,
                 sizeof(text),
                 "髋部/核心 %d | %.0f° | 保持稳定发力",
                 has_training ? snap.score_hip : 0,
                 (double)snap.hip_angle);
        lv_label_set_text(lbl_corr_hip, text);
    }
    if (lbl_corr_torso) {
        snprintf(text,
                 sizeof(text),
                 "躯干 %d | 前倾 %.0f° | 过大时抬胸收紧核心",
                 has_training ? snap.score_torso : 0,
                 (double)snap.torso_lean);
        lv_label_set_text(lbl_corr_torso, text);
    }
    if (lbl_corr_balance) {
        snprintf(text,
                 sizeof(text),
                 "左右平衡 %d | 膝角差 %.0f° | 差值越小越稳定",
                 has_training ? snap.score_balance : 0,
                 (double)snap.symmetry_delta);
        lv_label_set_text(lbl_corr_balance, text);
    }
    if (lbl_corr_track) {
        snprintf(text,
                 sizeof(text),
                 "识别稳定 %d | 有效点 %d | 人体尽量完整入镜",
                 has_training ? snap.score_track : 0,
                 snap.valid_kpts);
        lv_label_set_text(lbl_corr_track, text);
    }
    if (lbl_corr_tip) {
        snprintf(text,
                 sizeof(text),
                 "%s\n%s",
                 has_training ? snap.detail : "开始训练后显示当前动作的分项纠错。",
                 snap.cue);
        lv_label_set_text(lbl_corr_tip, text);
    }
}

static void update_training_ui(void)
{
    const training_snapshot_t snap = training_get_snapshot();
    if (!snap.profile) {
        return;
    }
    if (snap.state == TrainingRunState::Finished && !s_training_record_saved &&
        (snap.elapsed_ms > 0 || snap.count > 0)) {
        s_training_record_saved = true;
        training_record_save_snapshot(snap);
    }

    char text[160];
    if (lbl_train_subtitle) {
        char target_text[48];
        training_target_text_by_index(training_profile_index(snap.profile), target_text, sizeof(target_text));
        snprintf(text, sizeof(text), "%s · %s", snap.profile->name, target_text);
        lv_label_set_text(lbl_train_subtitle, text);
    }
    if (lbl_train_count) {
        if (snap.profile->counted) {
            snprintf(text, sizeof(text), "%d / %d", snap.count, snap.target_count);
        } else {
            snprintf(text, sizeof(text), "%s", training_state_text(snap.state));
        }
        lv_label_set_text(lbl_train_count, text);
    }
    if (lbl_train_timer) {
        char elapsed[16];
        format_mmss(snap.elapsed_ms, elapsed, sizeof(elapsed));
        if (!snap.profile->counted && training_goal_duration_for_index(training_profile_index(snap.profile)) > 0) {
            char target[16];
            format_mmss(training_goal_duration_for_index(training_profile_index(snap.profile)) * 1000U, target, sizeof(target));
            snprintf(text, sizeof(text), "%s / %s", elapsed, target);
        } else {
            snprintf(text, sizeof(text), "%s", elapsed);
        }
        lv_label_set_text(lbl_train_timer, text);
    }
    if (lbl_train_score) {
        snprintf(text, sizeof(text), "%d", snap.score);
        lv_label_set_text(lbl_train_score, text);
    }
    if (lbl_train_state) {
        snprintf(text, sizeof(text), "%s  第%d组\n点%d  PC %.1fFPS",
                 training_state_text(snap.state),
                 snap.current_set,
                 snap.valid_kpts,
                 (double)snap.pc_fps);
        lv_label_set_text(lbl_train_state, text);
    }
    if (lbl_train_cue) {
        lv_label_set_text(lbl_train_cue, snap.cue);
    }
    if (btn_train_primary_label) {
        const char *label = "开始";
        if (snap.state == TrainingRunState::Running) {
            label = "暂停";
        } else if (snap.state == TrainingRunState::Paused) {
            label = "继续";
        } else if (snap.state == TrainingRunState::Finished) {
            label = "重来";
        }
        lv_label_set_text(btn_train_primary_label, label);
    }
    if (lbl_start_current) {
        snprintf(text, sizeof(text), "当前: %s", snap.profile->name);
        lv_label_set_text(lbl_start_current, text);
    }
    if (lbl_start_target) {
        char target_text[48];
        training_target_text_by_index(training_profile_index(snap.profile), target_text, sizeof(target_text));
        snprintf(text, sizeof(text), "目标: %s", target_text);
        lv_label_set_text(lbl_start_target, text);
    }
    if (lbl_start_focus) {
        snprintf(text, sizeof(text), "重点: %s", snap.profile->focus);
        lv_label_set_text(lbl_start_focus, text);
    }
    if (lbl_start_sets_value) {
        snprintf(text, sizeof(text), "%d", training_goal_sets_for_index(training_profile_index(snap.profile)));
        lv_label_set_text(lbl_start_sets_value, text);
    }
    if (lbl_start_reps_title) {
        lv_label_set_text(lbl_start_reps_title, snap.profile->counted ? "每组次数" : "保持秒数");
    }
    if (lbl_start_reps_value) {
        if (snap.profile->counted) {
            snprintf(text, sizeof(text), "%d", training_goal_reps_for_index(training_profile_index(snap.profile)));
        } else {
            snprintf(text, sizeof(text), "%u", (unsigned)training_goal_duration_for_index(training_profile_index(snap.profile)));
        }
        lv_label_set_text(lbl_start_reps_value, text);
    }
    if (lbl_start_guide) {
        lv_label_set_text(lbl_start_guide, training_action_guide_for_index(training_profile_index(snap.profile)));
    }
    update_correction_ui_from_snapshot(snap);
}

static std::string lvgl_file_path_for_cover(const char *cover_path)
{
    if (!cover_path || !cover_path[0]) {
        return "";
    }
    std::string path = "S:";
    path += cover_path;
    return path;
}

static void update_music_ui(void)
{
    const music_state_t st = music_get_state_copy();
    if (lbl_music_title) {
        lv_label_set_text(lbl_music_title, st.title);
    }
    if (lbl_music_artist) {
        lv_label_set_text(lbl_music_artist, st.artist);
    }
    if (lbl_music_status) {
        lv_label_set_text(lbl_music_status, st.status);
        lv_obj_set_style_text_color(lbl_music_status,
                                    lv_color_hex(st.error ? 0xd64545 : (st.playing && !st.paused ? 0x22a06b : 0x687783)),
                                    LV_PART_MAIN);
    }
    if (lbl_music_time) {
        char pos[16];
        char dur[16];
        format_mmss(st.position_ms, pos, sizeof(pos));
        format_mmss(st.duration_ms, dur, sizeof(dur));
        char text[40];
        snprintf(text, sizeof(text), "%s / %s", pos, dur);
        lv_label_set_text(lbl_music_time, text);
    }
    if (bar_music_progress) {
        int32_t v = 0;
        if (st.duration_ms > 0) {
            v = (int32_t)(((uint64_t)st.position_ms * 1000ULL) / st.duration_ms);
            if (v > 1000) {
                v = 1000;
            }
        }
        lv_bar_set_value(bar_music_progress, v, LV_ANIM_OFF);
    }
    if (slider_music_volume) {
        lv_slider_set_value(slider_music_volume, st.volume, LV_ANIM_OFF);
    }
    if (slider_menu_volume) {
        lv_slider_set_value(slider_menu_volume, st.volume, LV_ANIM_OFF);
    }
    if (btn_music_play_label) {
        lv_label_set_text(btn_music_play_label, (st.playing && !st.paused) ? "暂停" : "播放");
    }
    if (lbl_music_cover_title) {
        lv_label_set_text(lbl_music_cover_title, st.title);
    }
    if (lbl_music_cover_path) {
        const char *source = st.library_ready ? "本地音乐 · TF卡曲库" : "等待TF卡曲库";
        lv_label_set_text(lbl_music_cover_path, source);
    }
    if (img_music_cover) {
        static char last_cover[160] = {};
        if (strcmp(last_cover, st.cover_path) != 0) {
            strlcpy(last_cover, st.cover_path, sizeof(last_cover));
            if (st.cover_path[0] && access(st.cover_path, F_OK) == 0) {
                static char lv_cover_path[192];
                std::string src = lvgl_file_path_for_cover(st.cover_path);
                strlcpy(lv_cover_path, src.c_str(), sizeof(lv_cover_path));
                lv_image_set_src(img_music_cover, lv_cover_path);
                lv_obj_clear_flag(img_music_cover, LV_OBJ_FLAG_HIDDEN);
            } else {
                lv_obj_add_flag(img_music_cover, LV_OBJ_FLAG_HIDDEN);
            }
        }
    }
    if (lbl_menu_music) {
        char text[96];
        snprintf(text, sizeof(text), "音乐: %s%s", st.playing ? (st.paused ? "暂停 " : "播放 ") : "", st.title);
        lv_label_set_text(lbl_menu_music, text);
    }
    if (lbl_pose_music) {
        char text[96];
        snprintf(text, sizeof(text), "%s%s", st.playing ? (st.paused ? "已暂停 " : "播放中 ") : "未播放 ", st.title);
        lv_label_set_text(lbl_pose_music, text);
    }
}

static void ui_timer_cb(lv_timer_t *timer)
{
    (void)timer;
    ui_msg_t msg;
    while (xQueueReceive(s_uiq, &msg, 0) == pdTRUE) {
        if (lbl_status) {
            lv_label_set_text(lbl_status, msg.line);
        }
        if (lbl_pose_status) {
            lv_label_set_text(lbl_pose_status, msg.line);
        }
        if (lbl_auth_status) {
            lv_label_set_text(lbl_auth_status, msg.line);
        }
        ESP_LOGI(TAG, "%s", msg.line);
    }
    if (s_auth_unlock_pending.load()) {
        auth_enter_main_ui();
    }
    update_preview_canvas_ui();
    update_face_overlay_ui();
    update_db_count_label();
    update_wifi_labels_ui();
    update_settings_status_ui();
    update_datetime_ui();
    update_music_ui();
    update_training_ui();
}

static void process_live_recognition(void)
{
    static char last_identity[80] = "人脸";

    if (!s_live_rec || !s_live_buf || !s_ai_buf || s_cam_w == 0 || s_cam_h == 0 || s_ai_w == 0 || s_ai_h == 0) {
        s_live_busy.store(false);
        return;
    }

    if (!lock_models(pdMS_TO_TICKS(50))) {
        s_live_busy.store(false);
        return;
    }
    if (!s_det || !s_rec) {
        release_pose_model();
        if (!ensure_face_models()) {
            unlock_models();
            s_live_busy.store(false);
            return;
        }
    }

    dl::image::img_t det_img = {};
    std::list<dl::detect::result_t> small_dets;
    const face_input_variant_t *variant = nullptr;
    if (!detect_face_with_variants(s_live_buf, &small_dets, &det_img, &variant)) {
        unlock_models();
        set_face_overlay(false, nullptr, nullptr);
        if ((s_frame_ctr % 20) == 0) {
            push_ui_msg("未检测到人脸");
        }
        s_live_busy.store(false);
        return;
    }

    std::list<dl::detect::result_t> full_dets =
        scale_detections(small_dets, s_ai_w, s_ai_h, s_cam_w, s_cam_h, variant ? variant->rotate_180 : false);
    dl::detect::result_t best_det = {};
    const bool have_best = get_largest_detection(full_dets, &best_det);

    char overlay_text[80] = "人脸";
    const int64_t now = esp_timer_get_time();
    const int64_t last_recognition = s_last_recognition_us.load();
    const bool should_recognize = s_rec->get_num_feats() > 0 &&
                                  (now - last_recognition >= LIVE_RECOGNITION_INTERVAL_US ||
                                   last_recognition == 0);
    if (should_recognize) {
        std::vector<dl::recognition::result_t> recs = s_rec->recognize(det_img, small_dets);
        s_last_recognition_us.store(now);

        if (recs.empty()) {
            snprintf(overlay_text, sizeof(overlay_text), "未知");
            snprintf(last_identity, sizeof(last_identity), "%s", overlay_text);
            push_ui_msg("检测到未知人脸");
        } else {
            const dl::recognition::result_t &res = recs.front();
            char name[48] = {};
            if (!load_face_name(res.id, name, sizeof(name))) {
                snprintf(name, sizeof(name), "id%u", (unsigned)res.id);
            }

            snprintf(overlay_text, sizeof(overlay_text), "%s %.2f", name, res.similarity);
            snprintf(last_identity, sizeof(last_identity), "%s", overlay_text);
            if (s_auth_face_login_active.load()) {
                strlcpy(s_auth_user, name, sizeof(s_auth_user));
                s_auth_face_login_active.store(false);
                s_auth_unlock_pending.store(true);
                s_live_rec = false;
            }
            push_ui_msg("%s 得分 %.3f", name, res.similarity);
        }
    } else if (s_rec->get_num_feats() == 0) {
        snprintf(overlay_text, sizeof(overlay_text), "人脸 / 无库");
    } else if (now - last_recognition <= FACE_OVERLAY_TIMEOUT_US) {
        snprintf(overlay_text, sizeof(overlay_text), "%s", last_identity);
    }

    unlock_models();
    if (have_best) {
        set_face_overlay(true, &best_det, overlay_text);
    }
    s_live_busy.store(false);
}

static void process_pose_detection(void)
{
    if (!s_pose_enabled || !s_pose_buf || s_cam_w == 0 || s_cam_h == 0 || s_pose_w == 0 || s_pose_h == 0) {
        s_pose_busy.store(false);
        return;
    }

    const int64_t total_start_us = esp_timer_get_time();
    (void)total_start_us;
    const uint32_t letterbox_x = s_local_pose_letterbox_x;
    const uint32_t letterbox_y = s_local_pose_letterbox_y;
    const uint32_t letterbox_w = s_local_pose_letterbox_w;
    const uint32_t letterbox_h = s_local_pose_letterbox_h;
    const bool use_letterbox = s_local_pose_use_letterbox;
    const uint32_t crop_x = s_local_pose_crop_x;
    const uint32_t crop_y = s_local_pose_crop_y;
    const uint32_t crop_w = s_local_pose_crop_w;
    const uint32_t crop_h = s_local_pose_crop_h;
    const bool use_crop = s_local_pose_use_crop;

    std::vector<pose_result_t> local_results;
    float local_pre_ms = 0.0f;
    float local_forward_ms = 0.0f;
    float local_post_ms = 0.0f;
    int local_valid_count = 0;
    float local_avg_score = 0.0f;
    float local_quality = 0.0f;
    bool local_ok = false;
    const char *local_model_name = "MoveNet";
    const bool pc_pose_available = CONFIG_PC_POSE_ENABLE && s_pc_pose_wifi_ready.load();
    const bool run_local_pose = LOCAL_POSE_ENABLE || (POSE_ENABLE_LOCAL_FALLBACK && !pc_pose_available);

    if (lock_models(pdMS_TO_TICKS(10))) {
        const int variant = (s_pose_input_variant >= 0 &&
                             s_pose_input_variant < (int)(sizeof(POSE_INPUT_VARIANTS) / sizeof(POSE_INPUT_VARIANTS[0])))
                                ? s_pose_input_variant
                                : 0;
        if (run_local_pose) {
            if (!s_pose_det) {
                ESP_LOGI(TAG, "Loading local MoveNet pose model...");
                s_pose_det = new HumanPoseDetect(0.18f, 0.70f, 1);
                if (s_pose_det && s_pose_det->ready()) {
                    ESP_LOGI(TAG, "Local MoveNet pose model ready");
                } else {
                    ESP_LOGE(TAG, "Local MoveNet pose model failed: %s", s_pose_det ? s_pose_det->error() : "alloc failed");
                }
            }

            if (s_pose_det && s_pose_det->ready()) {
                dl::image::img_t local_img = {};
                local_img.data = s_pose_buf;
                local_img.width = (uint16_t)s_pose_w;
                local_img.height = (uint16_t)s_pose_h;
                local_img.pix_type = POSE_INPUT_VARIANTS[variant].pix_type;

                const std::vector<pose_result_t> &detected = s_pose_det->run(local_img);
                if (use_crop && crop_w > 0 && crop_h > 0) {
                    local_results = scale_pose_results_from_crop(
                        detected,
                        s_pose_w,
                        s_pose_h,
                        crop_x,
                        crop_y,
                        crop_w,
                        crop_h,
                        s_cam_w,
                        s_cam_h,
                        POSE_INPUT_ROTATE_180);
                } else if (use_letterbox && letterbox_w > 0 && letterbox_h > 0) {
                    local_results = scale_pose_results_from_letterbox(
                        detected,
                        s_pose_w,
                        s_pose_h,
                        letterbox_x,
                        letterbox_y,
                        letterbox_w,
                        letterbox_h,
                        s_cam_w,
                        s_cam_h,
                        POSE_INPUT_ROTATE_180);
                } else {
                    local_results = scale_pose_results(detected, s_pose_w, s_pose_h, s_cam_w, s_cam_h, POSE_INPUT_ROTATE_180);
                }
                local_pre_ms = s_pose_det->last_preprocess_ms();
                local_forward_ms = s_pose_det->last_forward_ms();
                local_post_ms = s_pose_det->last_postprocess_ms();
                local_valid_count = s_pose_det->last_valid_count();
                local_avg_score = s_pose_det->last_avg_score();
                local_quality = s_pose_det->last_quality();
                local_ok = !local_results.empty();
                if (s_pose_input_variant != variant) {
                    s_pose_input_variant = variant;
                    push_ui_msg("Local pose input: %s", POSE_INPUT_VARIANTS[variant].name);
                }
                static int64_t last_local_input_log_us = 0;
                const int64_t input_log_us = esp_timer_get_time();
                if ((use_crop || use_letterbox) && input_log_us - last_local_input_log_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
                    last_local_input_log_us = input_log_us;
                    if (use_crop) {
                        ESP_LOGI(TAG,
                                 "Local pose input: body roi=(%u,%u,%ux%u)",
                                 (unsigned)crop_x,
                                 (unsigned)crop_y,
                                 (unsigned)crop_w,
                                 (unsigned)crop_h);
                    } else {
                        ESP_LOGI(TAG,
                                 "Local pose input: full letterbox=(%u,%u,%ux%u)",
                                 (unsigned)letterbox_x,
                                 (unsigned)letterbox_y,
                                 (unsigned)letterbox_w,
                                 (unsigned)letterbox_h);
                    }
                }
            }
        }
        unlock_models();
    }

    if (local_ok) {
        const bool draw_local = local_pose_draw_quality_ok(local_results, s_cam_w, s_cam_h);
        if (draw_local) {
            set_local_pose_overlay(local_results, s_cam_w, s_cam_h);
            update_pose_tracking_roi_from_pose(local_results[0], s_cam_w, s_cam_h, "local");
        } else {
            clear_local_pose_overlay();
            ESP_LOGW(TAG,
                     "Local %s pose weak, hide red overlay: valid=%d avg=%.3f q=%.1f",
                     local_model_name,
                     local_valid_count,
                     (double)local_avg_score,
                     (double)local_quality);
        }
        if (!local_results.empty() && !local_results[0].keypoints.empty()) {
            const auto &kp = local_results[0].keypoints[0];
            const auto &sh_l = local_results[0].keypoints.size() > 11 ? local_results[0].keypoints[11] : kp;
            const auto &sh_r = local_results[0].keypoints.size() > 12 ? local_results[0].keypoints[12] : kp;
            const auto &hip_l = local_results[0].keypoints.size() > 23 ? local_results[0].keypoints[23] : kp;
            const auto &hip_r = local_results[0].keypoints.size() > 24 ? local_results[0].keypoints[24] : kp;
            ESP_LOGI(TAG, "Local pose: people=%d nose=(%d,%d,%.2f) sh=(%d,%d,%.2f)/(%d,%d,%.2f) hip=(%d,%d,%.2f)/(%d,%d,%.2f) box=(%d,%d,%d,%d)",
                     (int)local_results.size(),
                     kp.x,
                     kp.y,
                     (double)kp.score,
                     sh_l.x,
                     sh_l.y,
                     (double)sh_l.score,
                     sh_r.x,
                     sh_r.y,
                     (double)sh_r.score,
                     hip_l.x,
                     hip_l.y,
                     (double)hip_l.score,
                     hip_r.x,
                     hip_r.y,
                     (double)hip_r.score,
                     local_results[0].x1,
                     local_results[0].y1,
                     local_results[0].x2,
                     local_results[0].y2);
        }
    } else {
        clear_local_pose_overlay();
        ESP_LOGW(TAG, "Local pose empty");
    }

    (void)pc_pose_available;

    const int64_t total_end_us = esp_timer_get_time();
    static int64_t fps_window_start_us = 0;
    static int fps_window_frames = 0;
    static float pose_fps = 0.0f;
    if (fps_window_start_us == 0) {
        fps_window_start_us = total_end_us;
    }
    fps_window_frames++;
    if (total_end_us - fps_window_start_us >= 1000000) {
        pose_fps = (float)fps_window_frames * 1000000.0f / (float)(total_end_us - fps_window_start_us);
        fps_window_frames = 0;
        fps_window_start_us = total_end_us;
    }

    static int64_t last_pose_msg_us = 0;
    if (total_end_us - last_pose_msg_us > 500000) {
        last_pose_msg_us = total_end_us;
        if (run_local_pose) {
            push_ui_msg("Local %s %.0f/%.0f/%.0f ms %d | PC async | %.1f FPS",
                        local_model_name,
                        (double)local_pre_ms,
                        (double)local_forward_ms,
                        (double)local_post_ms,
                        (int)local_results.size(),
                        (double)pose_fps);
        } else if (CONFIG_PC_POSE_ENABLE && !s_pc_pose_wifi_ready.load()) {
            push_ui_msg("PC Wi-Fi not ready | %.1f FPS", (double)pose_fps);
        } else if (CONFIG_PC_POSE_ENABLE && pc_pose_available) {
            push_ui_msg("PC pose async | %.1f FPS", (double)pose_fps);
        } else {
            push_ui_msg("Pose %.1f FPS", (double)pose_fps);
        }
    }

    s_pose_busy.store(false);
}

static void process_body_roi_detection(void)
{
    if (!s_pose_enabled || !s_body_det_buf || s_ai_w == 0 || s_ai_h == 0 || s_cam_w == 0 || s_cam_h == 0) {
        s_body_roi_busy.store(false);
        return;
    }

    if (!lock_models(pdMS_TO_TICKS(20))) {
        s_body_roi_busy.store(false);
        return;
    }

    bool reacquired = false;
    int det_count = 0;
    float best_score = 0.0f;
    int best_area = 0;
    const int64_t start_us = esp_timer_get_time();

    if (!s_body_det) {
        ESP_LOGI(TAG, "Loading pedestrian detector for body ROI...");
        s_body_det = new PedestrianDetect(PedestrianDetect::PICO_S8_V1, true);
        if (s_body_det) {
            s_body_det->set_score_thr(BODY_ROI_DETECT_SCORE_THR);
            s_body_det->set_nms_thr(BODY_ROI_DETECT_NMS_THR);
            ESP_LOGI(TAG, "Pedestrian detector ready");
        } else {
            ESP_LOGE(TAG, "Pedestrian detector allocation failed");
        }
    }

    if (s_body_det) {
        dl::image::img_t img = {};
        img.data = reinterpret_cast<uint16_t *>(s_body_det_buf);
        img.width = (uint16_t)s_ai_w;
        img.height = (uint16_t)s_ai_h;
        img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565LE;

        std::list<dl::detect::result_t> small_dets = s_body_det->run(img);
        std::list<dl::detect::result_t> dets = scale_detections(small_dets, s_ai_w, s_ai_h, s_cam_w, s_cam_h, false);
        dl::detect::result_t best_det = {};
        bool have_best = false;
        for (const auto &det : dets) {
            if (det.box.size() < 4 || det.score < BODY_ROI_DETECT_SCORE_THR) {
                continue;
            }
            const int area = detection_area(det);
            det_count++;
            if (!have_best || area > best_area) {
                best_det = det;
                best_area = area;
                best_score = det.score;
                have_best = true;
            }
        }
        if (have_best) {
            reacquired = update_pose_tracking_roi_from_detection(best_det, s_cam_w, s_cam_h, "pedestrian");
        }
    }

    const float total_ms = (float)(esp_timer_get_time() - start_us) / 1000.0f;
    static int64_t last_body_log_us = 0;
    const int64_t now_us = esp_timer_get_time();
    if (reacquired || now_us - last_body_log_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
        last_body_log_us = now_us;
        ESP_LOGI(TAG,
                 "Body ROI detect: %.0f ms dets=%d best=%.2f area=%d reacquired=%d",
                 (double)total_ms,
                 det_count,
                 (double)best_score,
                 best_area,
                 reacquired ? 1 : 0);
    }

    unlock_models();
    s_body_roi_busy.store(false);
}

static void process_pc_pose_frame(void)
{
    if (!s_pose_enabled || !s_pc_pose_buf || s_pose_w == 0 || s_pose_h == 0 || s_cam_w == 0 || s_cam_h == 0) {
        s_pc_pose_busy.store(false);
        return;
    }

    std::vector<pose_result_t> pc_results;
    float http_ms = 0.0f;
    float tx_ms = 0.0f;
    float wait_ms = 0.0f;
    float infer_ms = 0.0f;
    int valid_count = 0;
    const uint32_t crop_x = s_pc_pose_crop_x;
    const uint32_t crop_y = s_pc_pose_crop_y;
    const uint32_t crop_w = s_pc_pose_crop_w;
    const uint32_t crop_h = s_pc_pose_crop_h;
    const bool use_crop = s_pc_pose_use_crop;
    const uint32_t letterbox_x = s_pc_pose_letterbox_x;
    const uint32_t letterbox_y = s_pc_pose_letterbox_y;
    const uint32_t letterbox_w = s_pc_pose_letterbox_w;
    const uint32_t letterbox_h = s_pc_pose_letterbox_h;
    const bool use_letterbox = s_pc_pose_use_letterbox;
    static int pc_pose_miss_count = 0;

    const int64_t total_start_us = esp_timer_get_time();
    const bool pc_ok = run_pc_pose_request(
        s_pc_pose_buf, s_pose_w, s_pose_h, &pc_results, &http_ms, &tx_ms, &wait_ms, &infer_ms, &valid_count);

    const int64_t now_us = esp_timer_get_time();
    const float total_ms = (now_us - total_start_us) / 1000.0f;
    if (pc_ok) {
        if (use_letterbox) {
            pc_results = scale_pose_results_from_letterbox(
                pc_results,
                s_pose_w,
                s_pose_h,
                letterbox_x,
                letterbox_y,
                letterbox_w,
                letterbox_h,
                s_cam_w,
                s_cam_h,
                PC_POSE_SEND_ROTATE_180);
        } else if (use_crop) {
            pc_results = scale_pose_results_from_crop(
                pc_results, s_pose_w, s_pose_h, crop_x, crop_y, crop_w, crop_h, s_cam_w, s_cam_h, PC_POSE_SEND_ROTATE_180);
        } else {
            pc_results = scale_pose_results(pc_results, s_pose_w, s_pose_h, s_cam_w, s_cam_h, PC_POSE_SEND_ROTATE_180);
        }
        std::vector<pose_result_t> pc_results_raw = pc_results;
        std::vector<pose_result_t> pc_results_draw = pc_results_raw;
        if (!pc_results_raw.empty() && count_pc_pose_upper_visible(pc_results_raw[0]) >= PC_POSE_MIN_UPPER_KPTS) {
            smooth_pc_pose_results(pc_results_draw, s_cam_w, s_cam_h);
            pc_pose_miss_count = 0;
            s_last_pc_pose_success_us.store(now_us);
            s_body_roi_reacquire_requested.store(false);
            set_pc_pose_overlay(pc_results_draw, s_cam_w, s_cam_h);
            update_pose_tracking_roi_from_pose(pc_results_raw[0], s_cam_w, s_cam_h, "pc");
            pc_results = pc_results_raw;
        } else {
            note_pc_pose_miss(&pc_pose_miss_count);
            pc_results.clear();
        }
        static int64_t last_pc_crop_log_us = 0;
        const int64_t crop_log_us = esp_timer_get_time();
        if ((use_letterbox || use_crop) && crop_log_us - last_pc_crop_log_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
            last_pc_crop_log_us = crop_log_us;
            if (use_letterbox) {
                ESP_LOGI(TAG,
                         "PC pose input: full letterbox=(%u,%u,%ux%u)",
                         (unsigned)letterbox_x,
                         (unsigned)letterbox_y,
                         (unsigned)letterbox_w,
                         (unsigned)letterbox_h);
            } else {
                ESP_LOGI(TAG,
                         "PC pose input: body roi=(%u,%u,%ux%u)",
                         (unsigned)crop_x,
                         (unsigned)crop_y,
                         (unsigned)crop_w,
                         (unsigned)crop_h);
            }
        }
        static int64_t last_pc_pose_detail_log_us = 0;
        if (!pc_results.empty() && !pc_results[0].keypoints.empty() &&
            now_us - last_pc_pose_detail_log_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
            last_pc_pose_detail_log_us = now_us;
            const auto &kp = pc_results[0].keypoints[0];
            const auto &sh_l = pc_results[0].keypoints.size() > 11 ? pc_results[0].keypoints[11] : kp;
            const auto &sh_r = pc_results[0].keypoints.size() > 12 ? pc_results[0].keypoints[12] : kp;
            const auto &hip_l = pc_results[0].keypoints.size() > 23 ? pc_results[0].keypoints[23] : kp;
            const auto &hip_r = pc_results[0].keypoints.size() > 24 ? pc_results[0].keypoints[24] : kp;
            ESP_LOGI(TAG, "PC pose: people=%d nose=(%d,%d,%.2f) sh=(%d,%d,%.2f)/(%d,%d,%.2f) hip=(%d,%d,%.2f)/(%d,%d,%.2f) box=(%d,%d,%d,%d)",
                     (int)pc_results.size(),
                     kp.x,
                     kp.y,
                     (double)kp.score,
                     sh_l.x,
                     sh_l.y,
                     (double)sh_l.score,
                     sh_r.x,
                     sh_r.y,
                     (double)sh_r.score,
                     hip_l.x,
                     hip_l.y,
                     (double)hip_l.score,
                     hip_r.x,
                     hip_r.y,
                     (double)hip_r.score,
                     pc_results[0].x1,
                     pc_results[0].y1,
                     pc_results[0].x2,
                     pc_results[0].y2);
        }
    } else {
        note_pc_pose_miss(&pc_pose_miss_count);
        static int64_t last_pc_fail_log_us = 0;
        if (now_us - last_pc_fail_log_us >= PC_POSE_DETAIL_LOG_INTERVAL_US) {
            last_pc_fail_log_us = now_us;
            if (valid_count >= PC_POSE_MIN_VISIBLE_KPTS) {
                ESP_LOGW(TAG,
                         "PC pose board filter: hdr_valid=%d but parsed=0 (need>=%d kpts @%.2f)",
                         valid_count,
                         PC_POSE_MIN_VISIBLE_KPTS,
                         (double)POSE_OVERLAY_KPT_SCORE_THR);
            } else if (http_ms > 0.0f || wait_ms > 0.0f) {
                ESP_LOGW(TAG, "PC pose no person: wait=%.0fms infer=%.0fms hdr_valid=%d", wait_ms, infer_ms, valid_count);
            } else {
                ESP_LOGW(TAG, "PC pose TCP/send failed (no response timing)");
            }
        }
    }

    static int64_t fps_window_start_us = 0;
    static int fps_window_frames = 0;
    static float pc_fps = 0.0f;
    if (fps_window_start_us == 0) {
        fps_window_start_us = now_us;
    }
    fps_window_frames++;
    if (now_us - fps_window_start_us >= 1000000) {
        pc_fps = (float)fps_window_frames * 1000000.0f / (float)(now_us - fps_window_start_us);
        fps_window_frames = 0;
        fps_window_start_us = now_us;
    }

    if (pc_ok && !pc_results.empty()) {
        training_update_from_pc_pose(pc_results[0], valid_count, pc_fps);
    } else {
        training_lock();
        s_training_valid_kpts = valid_count;
        s_training_pc_fps = pc_fps;
        training_unlock();
    }

    static int64_t last_pc_msg_us = 0;
    if (now_us - last_pc_msg_us > 300000) {
        last_pc_msg_us = now_us;
        if (pc_ok) {
            push_ui_msg("PC %.1fFPS v%d %.0fms",
                        (double)pc_fps,
                        valid_count,
                        (double)total_ms);
        } else if (CONFIG_PC_POSE_ENABLE && !s_pc_pose_wifi_ready.load()) {
            push_ui_msg("PC wait %.1fFPS", (double)pc_fps);
        } else if (valid_count > 0) {
            push_ui_msg("PC drop%d %.1fFPS", valid_count, (double)pc_fps);
        } else if (http_ms > 0.0f) {
            push_ui_msg("PC none %.0fms %.1fFPS", (double)http_ms, (double)pc_fps);
        } else {
            push_ui_msg("PC net %.1fFPS", (double)pc_fps);
        }
    }

    s_pc_pose_busy.store(false);
}

static void process_enroll_job(void)
{
    if (!s_snap_buf || !s_snap_sem || s_cam_w == 0 || s_cam_h == 0 || s_enroll_name[0] == '\0') {
        push_ui_msg("摄像头或姓名未就绪");
        s_enroll_busy.store(false);
        return;
    }

    if (!lock_models(pdMS_TO_TICKS(1000))) {
        push_ui_msg("模型忙，正在重试...");
        s_enroll_busy.store(false);
        return;
    }
    if (!s_det || !s_rec) {
        release_pose_model();
        if (!ensure_face_models()) {
            unlock_models();
            push_ui_msg("人脸模型未就绪");
            s_enroll_busy.store(false);
            return;
        }
    }
    unlock_models();

    int attempt = 0;
    for (;;) {
        if (s_enroll_cancel.load()) {
            push_ui_msg("录入已取消");
            s_enroll_cancel.store(false);
            s_enroll_busy.store(false);
            return;
        }

        while (xSemaphoreTake(s_snap_sem, 0) == pdTRUE) {
        }

        attempt++;
        if (attempt == 1 || (attempt % 5) == 0) {
            push_ui_msg("正在寻找人脸... 第%d次", attempt);
        }

        s_snap_request.store(true);
        if (xSemaphoreTake(s_snap_sem, pdMS_TO_TICKS(2000)) != pdTRUE) {
            s_snap_request.store(false);
            push_ui_msg("等待摄像头画面...");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        if (!s_ai_buf || s_ai_w == 0 || s_ai_h == 0) {
            push_ui_msg("AI缓冲区未就绪");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        if (!lock_models(pdMS_TO_TICKS(1000))) {
            push_ui_msg("模型忙，正在重试...");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        if (!s_det || !s_rec) {
            unlock_models();
            push_ui_msg("人脸模型已释放");
            s_enroll_busy.store(false);
            return;
        }

        dl::image::img_t img = {};
        std::list<dl::detect::result_t> dets;
        const face_input_variant_t *variant = nullptr;
        if (!detect_face_with_variants(s_snap_buf, &dets, &img, &variant)) {
            unlock_models();
            vTaskDelay(pdMS_TO_TICKS(120));
            continue;
        }

        push_ui_msg("已检测到人脸，正在录入...");
        const int before = s_rec->get_num_feats();
        esp_err_t err = s_rec->enroll(img, dets);
        const int after = s_rec->get_num_feats();
        const uint16_t new_id = s_rec->get_last_feat_id();
        unlock_models();
        if (err != ESP_OK || after <= before) {
            push_ui_msg("录入失败，请保持稳定，正在重试...");
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }

        s_db_count.store(after);
        if (save_face_name(new_id, s_enroll_name) == ESP_OK) {
            push_ui_msg("已录入 %s (id=%u)", s_enroll_name, (unsigned)new_id);
            if (ta_name && bsp_display_lock(pdMS_TO_TICKS(50))) {
                lv_textarea_set_text(ta_name, "");
                bsp_display_unlock();
            }
        } else {
            push_ui_msg("人脸已录入，但姓名保存失败");
        }

        s_enroll_busy.store(false);
        return;
    }
}

static void face_worker_task(void *arg)
{
    (void)arg;

    for (;;) {
        bool did_work = false;

        if (s_enroll_sem && xSemaphoreTake(s_enroll_sem, 0) == pdTRUE) {
            did_work = true;
            process_enroll_job();
        }

        if (s_pose_sem && xSemaphoreTake(s_pose_sem, 0) == pdTRUE) {
            did_work = true;
            static int pose_wake_log_count = 0;
            if (pose_wake_log_count < 5) {
                ESP_LOGI(TAG, "Pose worker wake");
                pose_wake_log_count++;
            }
            process_pose_detection();
        }

        if (s_body_roi_sem && xSemaphoreTake(s_body_roi_sem, 0) == pdTRUE) {
            did_work = true;
            process_body_roi_detection();
        }

        if (s_live_sem && xSemaphoreTake(s_live_sem, 0) == pdTRUE) {
            did_work = true;
            process_live_recognition();
        }

        if (!did_work) {
            ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1000));
        } else {
            taskYIELD();
        }
    }
}

static void pc_pose_worker_task(void *arg)
{
    (void)arg;

    for (;;) {
        if (s_pc_pose_sem && xSemaphoreTake(s_pc_pose_sem, portMAX_DELAY) == pdTRUE) {
            process_pc_pose_frame();
        }
    }
}

typedef struct {
    char text[128];
    bool active;
} record_ui_update_t;

static void record_ui_update_async(void *arg)
{
    record_ui_update_t *update = static_cast<record_ui_update_t *>(arg);
    if (!update) {
        return;
    }
    if (s_record_status_label) {
        lv_label_set_text(s_record_status_label, update->text);
    }
    if (s_record_button) {
        set_button_text(s_record_button, update->active ? "停止录制" : "录制15秒");
    }
    delete update;
}

static void record_post_status(bool active, const char *fmt, ...)
{
    record_ui_update_t *update = new (std::nothrow) record_ui_update_t();
    if (!update) {
        return;
    }
    update->active = active;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(update->text, sizeof(update->text), fmt ? fmt : "", ap);
    va_end(ap);
    lv_async_call(record_ui_update_async, update);
}

static bool record_append_offline_manifest_locked(void)
{
    if (!s_record_path[0] || !s_record_action[0]) {
        return false;
    }
    FILE *fp = fopen(OFFLINE_VIDEO_MANIFEST, "a");
    if (!fp) {
        ESP_LOGW(TAG, "Record append manifest failed: %s errno=%d", OFFLINE_VIDEO_MANIFEST, errno);
        return false;
    }
    const training_profile_t *profile = training_profile_by_index(s_training_index);
    char id[48];
    snprintf(id, sizeof(id), "rec_%s_%lld", s_record_action, (long long)(esp_timer_get_time() / 1000000LL));
    fprintf(fp, "%s,录制-%s,%s,%s\n", id, profile->name, s_record_action, s_record_path);
    fclose(fp);
    return true;
}

static void record_stop_locked(bool completed, const char *reason)
{
    if (s_record_fp) {
        fclose(s_record_fp);
        s_record_fp = nullptr;
    }
    if (s_record_frame_buf) {
        heap_caps_free(s_record_frame_buf);
        s_record_frame_buf = nullptr;
    }
    const int frames = s_record_frames;
    bool appended = false;
    if (completed && frames > 0) {
        appended = record_append_offline_manifest_locked();
    } else if (s_record_path[0]) {
        unlink(s_record_path);
    }
    s_record_armed = false;
    s_recording = false;
    s_record_start_us = 0;
    s_record_last_frame_us = 0;
    s_record_frames = 0;
    if (completed && frames > 0) {
        record_post_status(false,
                           appended ? "录制完成：%d帧，已加入离线列表。" : "录制完成：%d帧，已保存到recordings。",
                           frames);
    } else {
        record_post_status(false, "%s", reason ? reason : "录制已取消。");
    }
}

static bool record_start_file_locked(int64_t now_us)
{
    mkdir(VIDEO_ROOT, 0775);
    mkdir(RECORD_VIDEO_DIR, 0775);
    if (!s_record_frame_buf) {
        s_record_frame_buf = (uint8_t *)heap_caps_malloc(RECORD_VIDEO_W * RECORD_VIDEO_H * 2,
                                                        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!s_record_frame_buf) {
            s_record_frame_buf = (uint8_t *)heap_caps_malloc(RECORD_VIDEO_W * RECORD_VIDEO_H * 2, MALLOC_CAP_8BIT);
        }
    }
    if (!s_record_frame_buf) {
        record_post_status(false, "录制失败：帧缓存不足。");
        s_record_armed = false;
        return false;
    }

    snprintf(s_record_path,
             sizeof(s_record_path),
             "%s/rec_%s_%lld.mjpeg",
             RECORD_VIDEO_DIR,
             s_record_action,
             (long long)(now_us / 1000LL));
    s_record_fp = fopen(s_record_path, "wb");
    if (!s_record_fp) {
        record_post_status(false, "录制失败：无法写入SD卡。");
        heap_caps_free(s_record_frame_buf);
        s_record_frame_buf = nullptr;
        s_record_armed = false;
        return false;
    }
    s_recording = true;
    s_record_start_us = now_us;
    s_record_last_frame_us = 0;
    s_record_frames = 0;
    record_post_status(true, "录制中：0/%d秒", RECORD_VIDEO_SECONDS);
    ESP_LOGI(TAG, "Record start: action=%s path=%s", s_record_action, s_record_path);
    return true;
}

static void record_process_frame(uint8_t *camera_buf, uint32_t w, uint32_t h, size_t camera_buf_len, int64_t now_us)
{
    if ((!s_record_armed && !s_recording) || !camera_buf || w == 0 || h == 0) {
        return;
    }
    if (!s_record_mutex || xSemaphoreTake(s_record_mutex, 0) != pdTRUE) {
        return;
    }

    if (!s_sdcard_mounted) {
        record_stop_locked(false, "录制停止：SD卡未挂载。");
        xSemaphoreGive(s_record_mutex);
        return;
    }

    if (s_record_armed && !s_recording) {
        const int64_t last_ok = s_last_pc_pose_success_us.load();
        if (last_ok <= 0 || now_us - last_ok > 900000) {
            xSemaphoreGive(s_record_mutex);
            return;
        }
        if (!record_start_file_locked(now_us)) {
            xSemaphoreGive(s_record_mutex);
            return;
        }
    }

    if (!s_recording || !s_record_fp || !s_record_frame_buf) {
        xSemaphoreGive(s_record_mutex);
        return;
    }

    if (now_us - s_record_start_us >= (int64_t)RECORD_VIDEO_SECONDS * 1000000LL) {
        record_stop_locked(true, "录制完成。");
        xSemaphoreGive(s_record_mutex);
        return;
    }

    const int64_t interval_us = 1000000LL / RECORD_VIDEO_FPS;
    if (s_record_last_frame_us > 0 && now_us - s_record_last_frame_us < interval_us) {
        xSemaphoreGive(s_record_mutex);
        return;
    }
    s_record_last_frame_us = now_us;

    if (!resize_rgb565_crop_ppa(s_record_frame_buf,
                                RECORD_VIDEO_W,
                                RECORD_VIDEO_H,
                                camera_buf,
                                w,
                                h,
                                camera_buf_len,
                                0,
                                0,
                                w,
                                h,
                                false)) {
        resize_rgb565_crop_nearest_strided(s_record_frame_buf,
                                           RECORD_VIDEO_W,
                                           RECORD_VIDEO_H,
                                           camera_buf,
                                           w,
                                           h,
                                           camera_buf_len,
                                           0,
                                           0,
                                           w,
                                           h,
                                           false);
    }
    dl::image::img_t img = {};
    img.data = s_record_frame_buf;
    img.width = RECORD_VIDEO_W;
    img.height = RECORD_VIDEO_H;
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565LE;
    dl::image::jpeg_img_t encoded = dl::image::sw_encode_jpeg(img, 72);
    if (encoded.data && encoded.data_len > 0) {
        fwrite(encoded.data, 1, encoded.data_len, s_record_fp);
        heap_caps_free(encoded.data);
        s_record_frames++;
        if ((s_record_frames % RECORD_VIDEO_FPS) == 0) {
            record_post_status(true,
                               "录制中：%d/%d秒",
                               s_record_frames / RECORD_VIDEO_FPS,
                               RECORD_VIDEO_SECONDS);
        }
    }
    xSemaphoreGive(s_record_mutex);
}

static void record_toggle_cb(lv_event_t *e)
{
    (void)e;
    if (!s_record_mutex) {
        s_record_mutex = xSemaphoreCreateMutex();
    }
    if (!s_record_mutex) {
        push_ui_msg("录制初始化失败：内存不足。");
        return;
    }
    if (xSemaphoreTake(s_record_mutex, pdMS_TO_TICKS(200)) != pdTRUE) {
        push_ui_msg("录制状态忙，请稍后。");
        return;
    }
    if (s_record_armed || s_recording) {
        record_stop_locked(false, "录制已取消。");
        xSemaphoreGive(s_record_mutex);
        return;
    }
    if (!s_sdcard_mounted) {
        xSemaphoreGive(s_record_mutex);
        push_ui_msg("SD卡未挂载，无法录制。");
        return;
    }
    strlcpy(s_record_action, training_action_id_from_index(s_training_index), sizeof(s_record_action));
    s_record_armed = true;
    s_recording = false;
    s_record_path[0] = '\0';
    s_record_frames = 0;
    record_post_status(true, "等待检测到人体后开始录制...");
    xSemaphoreGive(s_record_mutex);
}

static void video_frame_cb(uint8_t *camera_buf, uint8_t camera_buf_index, uint32_t w, uint32_t h,
                           size_t camera_buf_len)
{
    (void)camera_buf_index;
    const int64_t now_us = esp_timer_get_time();

    if (s_snap_request.load() && s_snap_buf && s_cam_buf_bytes > 0 && s_snap_sem) {
        copy_rgb565_compact(s_snap_buf, camera_buf, w, h, camera_buf_len);
        s_snap_request.store(false);
        xSemaphoreGive(s_snap_sem);
    }

    const int screen = g_ui_screen.load();
    lv_obj_t *canvas = nullptr;
    if (screen == static_cast<int>(UiScreen::Camera)) {
        canvas = cv_cam;
    } else if (screen == static_cast<int>(UiScreen::Auth)) {
        canvas = cv_auth;
    } else if (screen == static_cast<int>(UiScreen::Face)) {
        canvas = cv_face;
    } else if (screen == static_cast<int>(UiScreen::Pose)) {
        canvas = cv_pose;
    }

    const bool live_face_screen = screen == static_cast<int>(UiScreen::Face);
    const bool live_auth_screen = screen == static_cast<int>(UiScreen::Auth) && s_auth_face_login_active.load();
    if ((live_face_screen || live_auth_screen) && s_live_rec && !s_enroll_busy.load() && s_live_buf && s_live_sem) {
        s_frame_ctr++;
        if (!s_live_busy.exchange(true)) {
            copy_rgb565_compact(s_live_buf, camera_buf, w, h, camera_buf_len);
            xSemaphoreGive(s_live_sem);
            notify_worker();
        }
    } else if (screen == static_cast<int>(UiScreen::Pose) && s_pose_enabled) {
        pose_input_roi_t frame_roi = {};
        const bool have_frame_roi = get_pose_tracking_roi(w, h, &frame_roi);
        const int64_t last_body_roi_us = s_last_body_roi_request_us.load();
        const bool force_body_roi = s_body_roi_reacquire_requested.load();
        const int64_t last_pc_ok_us = s_last_pc_pose_success_us.load();
        const bool pc_pose_recent_ok = last_pc_ok_us > 0 && now_us - last_pc_ok_us <= BODY_ROI_SUPPRESS_AFTER_PC_OK_US;
        const bool need_body_roi = !have_frame_roi || (force_body_roi && !pc_pose_recent_ok);
        const int64_t body_roi_interval_us = BODY_ROI_DETECT_INTERVAL_US;
        if (need_body_roi && s_body_det_buf && s_body_roi_sem &&
            (last_body_roi_us == 0 || now_us - last_body_roi_us >= body_roi_interval_us) &&
            !s_body_roi_busy.exchange(true)) {
            resize_rgb565_crop_nearest_strided(
                s_body_det_buf, s_ai_w, s_ai_h, camera_buf, w, h, camera_buf_len, 0, 0, w, h, false);
            s_last_body_roi_request_us.store(now_us);
            s_body_roi_reacquire_requested.store(false);
            xSemaphoreGive(s_body_roi_sem);
            notify_worker();
        }

        const int64_t last_pc_pose_us = s_last_pc_pose_request_us.load();
        if (CONFIG_PC_POSE_ENABLE && s_pc_pose_wifi_ready.load() && s_pc_pose_buf && s_pc_pose_sem &&
            (last_pc_pose_us == 0 || now_us - last_pc_pose_us >= PC_POSE_FRAME_INTERVAL_US) &&
            !s_pc_pose_busy.exchange(true)) {
            pose_input_roi_t roi = {};
            if (have_frame_roi) {
                roi = frame_roi;
                s_pc_pose_crop_x = roi.x;
                s_pc_pose_crop_y = roi.y;
                s_pc_pose_crop_w = roi.w;
                s_pc_pose_crop_h = roi.h;
                s_pc_pose_use_crop = true;
                s_pc_pose_letterbox_x = 0;
                s_pc_pose_letterbox_y = 0;
                s_pc_pose_letterbox_w = 0;
                s_pc_pose_letterbox_h = 0;
                s_pc_pose_use_letterbox = false;
                if (!resize_rgb565_crop_ppa(
                        s_pc_pose_buf, s_pose_w, s_pose_h, camera_buf, w, h, camera_buf_len, roi.x, roi.y, roi.w, roi.h, PC_POSE_SEND_ROTATE_180)) {
                    resize_rgb565_crop_nearest_strided(
                        s_pc_pose_buf, s_pose_w, s_pose_h, camera_buf, w, h, camera_buf_len, roi.x, roi.y, roi.w, roi.h, PC_POSE_SEND_ROTATE_180);
                }
            } else {
                uint32_t box_x = 0;
                uint32_t box_y = 0;
                uint32_t box_w = s_pose_w;
                uint32_t box_h = s_pose_h;
                make_pose_letterbox_region(w, h, s_pose_w, s_pose_h, &box_x, &box_y, &box_w, &box_h);
                s_pc_pose_crop_x = 0;
                s_pc_pose_crop_y = 0;
                s_pc_pose_crop_w = w;
                s_pc_pose_crop_h = h;
                s_pc_pose_use_crop = false;
                s_pc_pose_letterbox_x = box_x;
                s_pc_pose_letterbox_y = box_y;
                s_pc_pose_letterbox_w = box_w;
                s_pc_pose_letterbox_h = box_h;
                s_pc_pose_use_letterbox = box_w > 0 && box_h > 0;
                resize_rgb565_letterbox_nearest_strided(
                    s_pc_pose_buf, s_pose_w, s_pose_h, camera_buf, w, h, camera_buf_len, box_x, box_y, box_w, box_h, PC_POSE_SEND_ROTATE_180);
            }
            s_last_pc_pose_request_us.store(now_us);
            xSemaphoreGive(s_pc_pose_sem);
            notify_pc_pose_worker();
        }

        const int64_t last_pose_us = s_last_pose_request_us.load();
        if (LOCAL_POSE_ENABLE && s_pose_sem &&
            s_pose_buf &&
            (last_pose_us == 0 || now_us - last_pose_us >= LOCAL_POSE_INTERVAL_US) && !s_pose_busy.exchange(true)) {
            pose_input_roi_t roi = {};
            if (have_frame_roi) {
                roi = frame_roi;
                s_local_pose_crop_x = roi.x;
                s_local_pose_crop_y = roi.y;
                s_local_pose_crop_w = roi.w;
                s_local_pose_crop_h = roi.h;
                s_local_pose_use_crop = true;
                s_local_pose_letterbox_x = 0;
                s_local_pose_letterbox_y = 0;
                s_local_pose_letterbox_w = 0;
                s_local_pose_letterbox_h = 0;
                s_local_pose_use_letterbox = false;
                if (!resize_rgb565_crop_ppa(
                        s_pose_buf, s_pose_w, s_pose_h, camera_buf, w, h, camera_buf_len, roi.x, roi.y, roi.w, roi.h, POSE_INPUT_ROTATE_180)) {
                    resize_rgb565_crop_nearest_strided(
                        s_pose_buf, s_pose_w, s_pose_h, camera_buf, w, h, camera_buf_len, roi.x, roi.y, roi.w, roi.h, POSE_INPUT_ROTATE_180);
                }
            } else {
                uint32_t box_x = 0;
                uint32_t box_y = 0;
                uint32_t box_w = s_pose_w;
                uint32_t box_h = s_pose_h;
                make_pose_letterbox_region(w, h, s_pose_w, s_pose_h, &box_x, &box_y, &box_w, &box_h);
                s_local_pose_crop_x = 0;
                s_local_pose_crop_y = 0;
                s_local_pose_crop_w = w;
                s_local_pose_crop_h = h;
                s_local_pose_use_crop = false;
                s_local_pose_letterbox_x = box_x;
                s_local_pose_letterbox_y = box_y;
                s_local_pose_letterbox_w = box_w;
                s_local_pose_letterbox_h = box_h;
                s_local_pose_use_letterbox = box_w > 0 && box_h > 0;
                resize_rgb565_letterbox_nearest_strided(
                    s_pose_buf, s_pose_w, s_pose_h, camera_buf, w, h, camera_buf_len, box_x, box_y, box_w, box_h, POSE_INPUT_ROTATE_180);
            }
            s_last_pose_request_us.store(now_us);
            xSemaphoreGive(s_pose_sem);
            notify_worker();
            static int pose_queue_log_count = 0;
            if (pose_queue_log_count < 5) {
                ESP_LOGI(TAG, "Pose request queued");
                pose_queue_log_count++;
            }
        }
    }

    if (screen == static_cast<int>(UiScreen::Pose)) {
        record_process_frame(camera_buf, w, h, camera_buf_len, now_us);
    }

    const int64_t last_preview_us = s_last_preview_update_us.load();
    if (canvas && s_preview_buf && (last_preview_us == 0 || now_us - last_preview_us >= 50000)) {
        s_last_preview_update_us.store(now_us);
        if (screen == static_cast<int>(UiScreen::Face) || screen == static_cast<int>(UiScreen::Auth)) {
            draw_face_overlay(camera_buf, w, h, camera_buf_len);
        } else if (screen == static_cast<int>(UiScreen::Pose)) {
            draw_pose_overlay(camera_buf, w, h, camera_buf_len);
        }
        if (!resize_rgb565_crop_ppa(s_preview_buf, PREVIEW_W, PREVIEW_H, camera_buf, w, h, camera_buf_len, 0, 0, w, h, false)) {
            compose_preview_rgb565(s_preview_buf, PREVIEW_W, PREVIEW_H, camera_buf, w, h, camera_buf_len);
        }
        s_preview_dirty.store(true);
    }
}

static esp_err_t start_stream(void)
{
    if (s_vfd < 0) {
        return ESP_ERR_INVALID_STATE;
    }
    if (s_streaming) {
        return ESP_OK;
    }

    ESP_RETURN_ON_ERROR(app_video_stream_task_start(s_vfd, 0), TAG, "stream start");
    s_streaming = true;
    return ESP_OK;
}

static esp_err_t ensure_frame_buffers(void)
{
    if (!s_snap_sem) {
        s_snap_sem = xSemaphoreCreateBinary();
        if (!s_snap_sem) {
            return ESP_ERR_NO_MEM;
        }
    }

    if (!s_live_sem) {
        s_live_sem = xSemaphoreCreateBinary();
        if (!s_live_sem) {
            return ESP_ERR_NO_MEM;
        }
    }

    if (!s_snap_buf && s_cam_buf_bytes > 0) {
        s_snap_buf = (uint8_t *)heap_caps_malloc(s_cam_buf_bytes, MALLOC_CAP_SPIRAM);
        if (!s_snap_buf) {
            return ESP_ERR_NO_MEM;
        }
    }

    if (!s_live_buf && s_cam_buf_bytes > 0) {
        s_live_buf = (uint8_t *)heap_caps_malloc(s_cam_buf_bytes, MALLOC_CAP_SPIRAM);
        if (!s_live_buf) {
            return ESP_ERR_NO_MEM;
        }
    }

    if (!s_preview_buf) {
        s_preview_buf = (uint8_t *)heap_caps_malloc((size_t)PREVIEW_W * (size_t)PREVIEW_H * 2,
                                                   MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
        if (!s_preview_buf) {
            s_preview_buf = (uint8_t *)heap_caps_malloc((size_t)PREVIEW_W * (size_t)PREVIEW_H * 2, MALLOC_CAP_SPIRAM);
        }
        if (!s_preview_buf) {
            return ESP_ERR_NO_MEM;
        }
        memset(s_preview_buf, 0, (size_t)PREVIEW_W * (size_t)PREVIEW_H * 2);
        ESP_LOGI(TAG, "Preview frame: %" PRId32 " x %" PRId32, PREVIEW_W, PREVIEW_H);
    }

    if (s_ai_w == 0 || s_ai_h == 0) {
        s_ai_w = std::min<uint32_t>(AI_FRAME_W, s_cam_w);
        s_ai_h = std::min<uint32_t>(AI_FRAME_H, s_cam_h);
    }

    if (s_pose_w == 0 || s_pose_h == 0) {
        s_pose_w = std::min<uint32_t>(POSE_FRAME_W, s_cam_w);
        s_pose_h = std::min<uint32_t>(POSE_FRAME_H, s_cam_h);
    }

    if (!s_ai_buf && s_ai_w > 0 && s_ai_h > 0) {
        s_ai_buf = (uint8_t *)heap_caps_malloc((size_t)s_ai_w * (size_t)s_ai_h * 2, MALLOC_CAP_SPIRAM);
        if (!s_ai_buf) {
            return ESP_ERR_NO_MEM;
        }
        ESP_LOGI(TAG, "AI detection frame: %" PRIu32 " x %" PRIu32, s_ai_w, s_ai_h);
    }

    if (!s_body_det_buf && s_ai_w > 0 && s_ai_h > 0) {
        s_body_det_buf =
            (uint8_t *)heap_caps_malloc((size_t)s_ai_w * (size_t)s_ai_h * 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
        if (!s_body_det_buf) {
            s_body_det_buf = (uint8_t *)heap_caps_malloc((size_t)s_ai_w * (size_t)s_ai_h * 2, MALLOC_CAP_SPIRAM);
        }
        if (!s_body_det_buf) {
            return ESP_ERR_NO_MEM;
        }
        ESP_LOGI(TAG, "Body ROI detection frame: %" PRIu32 " x %" PRIu32, s_ai_w, s_ai_h);
    }

    if (!s_pose_buf && s_pose_w > 0 && s_pose_h > 0) {
        const size_t pose_bytes = (size_t)s_pose_w * (size_t)s_pose_h * 2;
        s_pose_buf = (uint8_t *)heap_caps_malloc(pose_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        const char *pose_mem = "internal";
        if (!s_pose_buf) {
            s_pose_buf = (uint8_t *)heap_caps_malloc(pose_bytes, MALLOC_CAP_SPIRAM);
            pose_mem = "psram";
        }
        if (!s_pose_buf) {
            return ESP_ERR_NO_MEM;
        }
        ESP_LOGI(TAG, "Pose source frame: %" PRIu32 " x %" PRIu32 " (%s)", s_pose_w, s_pose_h, pose_mem);
    }

    if (!s_pc_pose_buf && s_pose_w > 0 && s_pose_h > 0) {
        const size_t pose_bytes = (size_t)s_pose_w * (size_t)s_pose_h * 2;
        const size_t align = s_cache_line > 0 ? s_cache_line : 64;
        s_pc_pose_buf = (uint8_t *)heap_caps_aligned_calloc(align, 1, pose_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
        if (!s_pc_pose_buf) {
            s_pc_pose_buf = (uint8_t *)heap_caps_aligned_calloc(align, 1, pose_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        }
        if (!s_pc_pose_buf) {
            s_pc_pose_buf = (uint8_t *)heap_caps_aligned_calloc(align, 1, pose_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT | MALLOC_CAP_DMA);
        }
        if (!s_pc_pose_buf) {
            return ESP_ERR_NO_MEM;
        }
        ESP_LOGI(TAG, "PC pose source frame: %" PRIu32 " x %" PRIu32 " align=%u", s_pose_w, s_pose_h, (unsigned)align);
    }

    return ESP_OK;
}

static esp_err_t init_camera_pipeline(void)
{
    if (s_vfd >= 0) {
        return start_stream();
    }

    i2c_master_bus_handle_t i2c = bsp_i2c_get_handle();
    esp_err_t video_err = app_video_main(i2c);
    if (video_err != ESP_OK) {
        ESP_LOGE(TAG,
                 "app_video_main failed: %s psram=%u internal=%u largest_psram=%u largest_internal=%u",
                 esp_err_to_name(video_err),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
        return video_err;
    }

    s_vfd = app_video_open((char *)EXAMPLE_CAM_DEV_PATH, APP_VIDEO_FMT_RGB565);
    if (s_vfd < 0) {
        ESP_LOGE(TAG, "app_video_open failed");
        return ESP_FAIL;
    }

    app_video_get_resolution(&s_cam_w, &s_cam_h);
    ESP_LOGI(TAG, "Camera resolution: %" PRIu32 " x %" PRIu32, s_cam_w, s_cam_h);

    ESP_ERROR_CHECK(esp_cache_get_alignment(MALLOC_CAP_SPIRAM, &s_cache_line));
    s_cam_buf_bytes = (size_t)s_cam_w * (size_t)s_cam_h * 2;
    for (int i = 0; i < EXAMPLE_CAM_BUF_NUM; i++) {
        s_cam_buf[i] = (uint8_t *)heap_caps_aligned_alloc(s_cache_line, s_cam_buf_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_DMA | MALLOC_CAP_8BIT);
        if (!s_cam_buf[i]) {
            s_cam_buf[i] = (uint8_t *)heap_caps_aligned_alloc(s_cache_line, s_cam_buf_bytes, MALLOC_CAP_SPIRAM);
        }
        if (!s_cam_buf[i]) {
            return ESP_ERR_NO_MEM;
        }
    }

    ESP_RETURN_ON_ERROR(ensure_frame_buffers(), TAG, "frame buffers");
    ESP_RETURN_ON_ERROR(app_video_set_bufs(s_vfd, EXAMPLE_CAM_BUF_NUM, (const void **)s_cam_buf), TAG, "set bufs");
    ESP_RETURN_ON_ERROR(app_video_register_frame_operation_cb(video_frame_cb), TAG, "register frame cb");
    ESP_RETURN_ON_ERROR(start_stream(), TAG, "start stream");

    if (bsp_display_lock(50)) {
        place_canvas(cv_auth, view_auth, PREVIEW_W, PREVIEW_H);
        place_canvas(cv_cam, view_cam, PREVIEW_W, PREVIEW_H);
        place_canvas(cv_face, view_face, PREVIEW_W, PREVIEW_H);
        place_canvas(cv_pose, view_pose, PREVIEW_W, PREVIEW_H);
        bsp_display_unlock();
    }

    return ESP_OK;
}

static void reset_page_refs(lv_obj_t **screen)
{
    if (screen == &scr_auth) {
        view_auth = nullptr;
        cv_auth = nullptr;
        lbl_auth_status = nullptr;
        s_auth_brand = nullptr;
        ta_auth_name = nullptr;
        ta_auth_pass = nullptr;
        kb_auth = nullptr;
    } else if (screen == &scr_profile) {
        ta_profile_name = nullptr;
        ta_profile_height = nullptr;
        ta_profile_weight = nullptr;
        ta_profile_minutes = nullptr;
        lbl_profile_choice = nullptr;
        lbl_profile_status = nullptr;
        kb_profile = nullptr;
    } else if (screen == &scr_menu) {
        lbl_menu_wifi = nullptr;
        lbl_menu_pc = nullptr;
        lbl_menu_ip = nullptr;
        lbl_menu_music = nullptr;
        lbl_menu_datetime = nullptr;
        lbl_menu_weather = nullptr;
        slider_menu_volume = nullptr;
    } else if (screen == &scr_cam) {
        view_cam = nullptr;
        cv_cam = nullptr;
    } else if (screen == &scr_pose) {
        view_pose = nullptr;
        cv_pose = nullptr;
        lbl_pose_status = nullptr;
        lbl_pose_music = nullptr;
        lbl_train_subtitle = nullptr;
        lbl_train_count = nullptr;
        lbl_train_timer = nullptr;
        lbl_train_score = nullptr;
        lbl_train_state = nullptr;
        lbl_train_cue = nullptr;
        btn_train_primary_label = nullptr;
        s_record_button = nullptr;
        s_record_status_label = nullptr;
    } else if (screen == &scr_face) {
        box_face = nullptr;
        lbl_face_name = nullptr;
        lbl_db_count = nullptr;
        ta_name = nullptr;
        ta_del_id = nullptr;
        kb_name = nullptr;
        sw_live = nullptr;
        lbl_status = nullptr;
    } else if (screen == &scr_settings) {
        lbl_wifi_state = nullptr;
        lbl_wifi_ip = nullptr;
        lbl_wifi_saved_ssid = nullptr;
        lbl_settings_account = nullptr;
        lbl_settings_face = nullptr;
        lbl_settings_profile = nullptr;
        lbl_settings_heap = nullptr;
        lbl_settings_uptime = nullptr;
        lbl_settings_pc = nullptr;
        ta_wifi_ssid = nullptr;
        ta_wifi_pass = nullptr;
        kb_settings = nullptr;
    } else if (screen == &scr_music) {
        lbl_music_title = nullptr;
        lbl_music_artist = nullptr;
        lbl_music_status = nullptr;
        lbl_music_time = nullptr;
        lbl_music_cover_title = nullptr;
        lbl_music_cover_path = nullptr;
        img_music_cover = nullptr;
        bar_music_progress = nullptr;
        slider_music_volume = nullptr;
        btn_music_play_label = nullptr;
    } else if (screen == &scr_start) {
        lbl_start_current = nullptr;
        lbl_start_target = nullptr;
        lbl_start_focus = nullptr;
        lbl_start_sets_value = nullptr;
        lbl_start_reps_title = nullptr;
        lbl_start_reps_value = nullptr;
        lbl_start_guide = nullptr;
    } else if (screen == &scr_plan) {
        panel_plan_today = nullptr;
        panel_plan_week = nullptr;
        btn_plan_week_label = nullptr;
    } else if (screen == &scr_correction) {
        lbl_corr_summary = nullptr;
        lbl_corr_depth = nullptr;
        lbl_corr_knee = nullptr;
        lbl_corr_hip = nullptr;
        lbl_corr_torso = nullptr;
        lbl_corr_balance = nullptr;
        lbl_corr_track = nullptr;
        lbl_corr_tip = nullptr;
        s_offline_player_card = nullptr;
        s_offline_player_host = nullptr;
        s_offline_list = nullptr;
        s_offline_status_label = nullptr;
        s_offline_report_label = nullptr;
        s_offline_progress = nullptr;
        s_offline_title_label = nullptr;
        s_offline_zoom_button = nullptr;
    } else if (screen == &scr_courses) {
        s_course_player_card = nullptr;
        s_course_player_host = nullptr;
        s_course_list = nullptr;
        s_course_status_label = nullptr;
        s_course_title_label = nullptr;
        s_course_zoom_button = nullptr;
    }
}

static bool create_page_checked(lv_obj_t **screen, create_product_page_fn_t create_fn, const char *name)
{
    if (!screen || !create_fn) {
        return false;
    }
    if (*screen) {
        return true;
    }

    s_lvgl_build_failed = false;
    create_fn();
    if (!*screen || s_lvgl_build_failed) {
        ESP_LOGW(TAG,
                 "UI page %s create failed; will retry. psram=%u internal=%u",
                 name ? name : "?",
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
        if (*screen) {
            lv_obj_delete(*screen);
            *screen = nullptr;
        }
        reset_page_refs(screen);
        s_lvgl_build_failed = false;
        return false;
    }

    ESP_LOGI(TAG, "UI page %s ready", name ? name : "?");
    return true;
}

static bool preload_page_by_index(int index)
{
    switch (index) {
    case 0:
        return create_page_checked(&scr_start, create_start_page, "start");
    case 1:
        return create_page_checked(&scr_correction, create_correction_page, "correction");
    case 2:
        return create_page_checked(&scr_courses, create_courses_page, "courses");
    case 3:
        return create_page_checked(&scr_plan, create_plan_page, "plan");
    case 4:
        return create_page_checked(&scr_report, create_report_page, "report");
    case 5:
        return create_page_checked(&scr_music, create_music_page, "music");
    case 6:
        return create_page_checked(&scr_settings, create_settings_page, "settings");
    case 7:
        return create_page_checked(&scr_pose, create_pose_page, "pose");
    default:
        return true;
    }
}

static bool preload_page_ready_by_index(int index)
{
    switch (index) {
    case 0:
        return scr_start != nullptr;
    case 1:
        return scr_correction != nullptr;
    case 2:
        return scr_courses != nullptr;
    case 3:
        return scr_plan != nullptr;
    case 4:
        return scr_report != nullptr;
    case 5:
        return scr_music != nullptr;
    case 6:
        return scr_settings != nullptr;
    case 7:
        return scr_pose != nullptr;
    default:
        return true;
    }
}

static void page_preload_timer_cb(lv_timer_t *timer)
{
    static constexpr int PAGE_COUNT = 8;

    if (s_pending_screen) {
        if (create_page_checked(s_pending_screen, s_pending_create_fn, s_pending_page_name)) {
            const bool start_pose_after_load = s_pending_start_pose && s_pending_screen == &scr_pose;
            lv_scr_load(*s_pending_screen);
            ESP_LOGI(TAG, "Pending UI page %s loaded", s_pending_page_name ? s_pending_page_name : "?");
            s_pending_screen = nullptr;
            s_pending_create_fn = nullptr;
            s_pending_page_name = nullptr;
            s_pending_start_pose = false;
            if (start_pose_after_load && init_camera_pipeline() != ESP_OK) {
                push_ui_msg("摄像头启动失败，请检查排线和传感器配置。");
            }
        }
        return;
    }

    for (int attempts = 0; attempts < PAGE_COUNT; attempts++) {
        const int index = (s_page_preload_index + attempts) % PAGE_COUNT;
        if (!preload_page_ready_by_index(index)) {
            if (preload_page_by_index(index)) {
                s_page_preload_index = (index + 1) % PAGE_COUNT;
            } else {
                s_page_preload_index = index;
            }
            return;
        }
    }

    if (!s_page_preload_complete) {
        ESP_LOGI(TAG, "UI page preload complete");
        s_page_preload_complete = true;
    }
    if (timer) {
        lv_timer_pause(timer);
    }
}

static void start_page_preloader(void)
{
    s_page_preload_complete = false;
    if (s_page_preload_timer) {
        lv_timer_resume(s_page_preload_timer);
        lv_timer_ready(s_page_preload_timer);
        return;
    }
    s_page_preload_timer = lv_timer_create(page_preload_timer_cb, 80, nullptr);
    if (s_page_preload_timer) {
        lv_timer_ready(s_page_preload_timer);
    } else {
        mark_lvgl_build_failed("page preload timer");
    }
}

static bool preload_all_product_pages_now(void)
{
    bool ok = true;
    ok = create_page_checked(&scr_start, create_start_page, "start") && ok;
    ok = create_page_checked(&scr_correction, create_correction_page, "correction") && ok;
    ok = create_page_checked(&scr_courses, create_courses_page, "courses") && ok;
    ok = create_page_checked(&scr_plan, create_plan_page, "plan") && ok;
    ok = create_page_checked(&scr_music, create_music_page, "music") && ok;
    ok = create_page_checked(&scr_report, create_report_page, "report") && ok;
    ok = create_page_checked(&scr_settings, create_settings_page, "settings") && ok;
    ok = create_page_checked(&scr_pose, create_pose_page, "pose") && ok;

    if (ok) {
        s_page_preload_complete = true;
        ESP_LOGI(TAG, "UI page preload complete");
    }
    return ok;
}

static void training_card_cb(lv_event_t *e)
{
    const int index = (int)(intptr_t)lv_event_get_user_data(e);
    training_select_index(index);
    update_training_ui();
    const training_profile_t *profile = training_profile_by_index(index);
    push_ui_msg("已选择%s，可调整目标后开始。", profile->name);
}

static void training_start_selected_cb(lv_event_t *e)
{
    training_start_or_resume();
    show_pose_cb(e);
}

static void training_goal_adjust_cb(lv_event_t *e)
{
    const int code = (int)(intptr_t)lv_event_get_user_data(e);
    training_lock();
    const int index = std::max(0, std::min(TRAINING_PROFILE_COUNT - 1, s_training_index));
    const training_profile_t *profile = training_profile_by_index(index);
    if (code == 1 || code == -1) {
        if (profile->counted) {
            int next_sets = training_goal_sets_for_index(index) + code;
            next_sets = std::max(1, std::min(9, next_sets));
            s_training_goal_sets[index] = (uint8_t)next_sets;
        }
    } else if (code == 2 || code == -2) {
        if (profile->counted) {
            int next_reps = training_goal_reps_for_index(index) + (code > 0 ? 1 : -1);
            next_reps = std::max(1, std::min(99, next_reps));
            s_training_goal_reps[index] = (uint8_t)next_reps;
        } else {
            int next_sec = (int)training_goal_duration_for_index(index) + (code > 0 ? 15 : -15);
            next_sec = std::max(15, std::min(1800, next_sec));
            s_training_goal_duration_sec[index] = (uint16_t)next_sec;
        }
    }
    training_reset_locked();
    training_unlock();

    esp_err_t err = save_training_goals();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "save training goals failed: %s", esp_err_to_name(err));
    }
    update_training_ui();
}

static void plan_toggle_week_cb(lv_event_t *e)
{
    (void)e;
    if (!panel_plan_today || !panel_plan_week) {
        return;
    }
    const bool week_hidden = lv_obj_has_flag(panel_plan_week, LV_OBJ_FLAG_HIDDEN);
    if (week_hidden) {
        lv_obj_add_flag(panel_plan_today, LV_OBJ_FLAG_HIDDEN);
        lv_obj_clear_flag(panel_plan_week, LV_OBJ_FLAG_HIDDEN);
        if (btn_plan_week_label) {
            lv_label_set_text(btn_plan_week_label, "今日计划");
        }
    } else {
        lv_obj_clear_flag(panel_plan_today, LV_OBJ_FLAG_HIDDEN);
        lv_obj_add_flag(panel_plan_week, LV_OBJ_FLAG_HIDDEN);
        if (btn_plan_week_label) {
            lv_label_set_text(btn_plan_week_label, "查看本周");
        }
    }
}

static int current_weekday_monday_index(void)
{
    time_t now = time(nullptr);
    if (now < 1700000000) {
        return -1;
    }
    struct tm tm_now = {};
    localtime_r(&now, &tm_now);
    if (tm_now.tm_wday < 0 || tm_now.tm_wday > 6) {
        return -1;
    }
    return (tm_now.tm_wday + 6) % 7;
}

static bool training_record_is_today(void)
{
    time_t now = time(nullptr);
    if (now < 1700000000 || s_training_record.last_time[0] == '\0') {
        return false;
    }
    struct tm tm_now = {};
    localtime_r(&now, &tm_now);
    char today[16];
    snprintf(today,
             sizeof(today),
             "%04d-%02d-%02d",
             tm_now.tm_year + 1900,
             tm_now.tm_mon + 1,
             tm_now.tm_mday);
    return strncmp(s_training_record.last_time, today, strlen(today)) == 0;
}

static int training_record_progress_for_index(int index)
{
    if (index < 0 || index >= TRAINING_PROFILE_COUNT ||
        s_training_record.sessions == 0 ||
        !training_record_is_today()) {
        return 0;
    }
    const training_profile_t *profile = training_profile_by_index(index);
    if (strcmp(s_training_record.last_training, profile->name) != 0) {
        return 0;
    }
    if (profile->counted) {
        const int target = std::max(1, training_target_count(profile));
        return std::max(0, std::min(100, s_training_record.last_count * 100 / target));
    }
    const uint32_t target_ms = std::max<uint32_t>(1, training_goal_duration_for_index(index) * 1000U);
    return std::max(0, std::min(100, (int)(s_training_record.last_elapsed_ms * 100U / target_ms)));
}

static void training_toggle_cb(lv_event_t *e)
{
    (void)e;
    training_toggle_session();
    update_training_ui();
}

static void training_finish_cb(lv_event_t *e)
{
    (void)e;
    training_finish_session();
    update_training_ui();
}

static void show_menu_cb(lv_event_t *e)
{
    (void)e;
    ESP_LOGI(TAG, "Show menu page");
    video_player_stop_current();
    training_pause_if_running();
    s_live_rec = false;
    s_auth_face_login_active.store(false);
    s_live_busy.store(false);
    s_pose_enabled = false;
    s_pose_busy.store(false);
    s_pc_pose_busy.store(false);
    s_enroll_cancel.store(true);
    set_face_overlay(false, nullptr, nullptr);
    clear_pose_overlay();
    s_last_recognition_us.store(0);
    s_last_pose_request_us.store(0);
    s_last_pc_pose_request_us.store(0);
    if (sw_live) {
        lv_obj_clear_state(sw_live, LV_STATE_CHECKED);
    }
    g_ui_screen.store(static_cast<int>(UiScreen::Menu));
    if (kb_name) {
        lv_obj_add_flag(kb_name, LV_OBJ_FLAG_HIDDEN);
        lv_keyboard_set_textarea(kb_name, nullptr);
    }
    if (kb_auth) {
        lv_obj_add_flag(kb_auth, LV_OBJ_FLAG_HIDDEN);
        lv_keyboard_set_textarea(kb_auth, nullptr);
    }
    if (kb_settings) {
        lv_obj_add_flag(kb_settings, LV_OBJ_FLAG_HIDDEN);
        lv_keyboard_set_textarea(kb_settings, nullptr);
    }
    release_models_if_idle(true, true);

    if (bsp_display_lock(portMAX_DELAY)) {
        destroy_page_if_inactive(&scr_menu);
        create_page_checked(&scr_menu, create_menu, "overview");
        lv_scr_load(scr_menu);
        bsp_display_unlock();
    }
}

static void show_cam_cb(lv_event_t *e)
{
    (void)e;
    ESP_LOGI(TAG, "Show camera page");
    video_player_stop_current();
    s_enroll_cancel.store(true);
    s_live_rec = false;
    s_auth_face_login_active.store(false);
    s_live_busy.store(false);
    s_pose_enabled = false;
    s_pose_busy.store(false);
    s_pc_pose_busy.store(false);
    set_face_overlay(false, nullptr, nullptr);
    clear_pose_overlay();
    s_last_recognition_us.store(0);
    s_last_pose_request_us.store(0);
    s_last_pc_pose_request_us.store(0);
    if (sw_live) {
        lv_obj_clear_state(sw_live, LV_STATE_CHECKED);
    }
    release_models_if_idle(true, true);
    g_ui_screen.store(static_cast<int>(UiScreen::Camera));

    if (bsp_display_lock(portMAX_DELAY)) {
        if (create_page_checked(&scr_cam, create_camera_page, "camera")) {
            lv_scr_load(scr_cam);
        }
        bsp_display_unlock();
    }

    if (init_camera_pipeline() != ESP_OK) {
        push_ui_msg("摄像头启动失败，请检查排线和传感器配置。");
    }
}

static void show_face_cb(lv_event_t *e)
{
    (void)e;
    ESP_LOGI(TAG, "Show face page");
    video_player_stop_current();
    s_auth_face_login_active.store(false);
    s_pose_enabled = false;
    s_pose_busy.store(false);
    s_pc_pose_busy.store(false);
    s_body_roi_busy.store(false);
    s_last_body_roi_request_us.store(0);
    s_body_roi_reacquire_requested.store(false);
    s_last_pose_request_us.store(0);
    s_last_pc_pose_request_us.store(0);
    clear_pose_overlay();
    release_models_if_idle(true, false);
    g_ui_screen.store(static_cast<int>(UiScreen::Face));
    push_ui_msg("人脸页面就绪");

    if (bsp_display_lock(portMAX_DELAY)) {
        if (create_page_checked(&scr_face, create_face_page, "face")) {
            lv_scr_load(scr_face);
        }
        bsp_display_unlock();
    }

    if (init_camera_pipeline() != ESP_OK) {
        push_ui_msg("摄像头启动失败，请检查排线和传感器配置。");
    }
}

static void show_pose_cb(lv_event_t *e)
{
    (void)e;
    ESP_LOGI(TAG, "Show pose page");
    video_player_stop_current();
    s_enroll_cancel.store(true);
    s_live_rec = false;
    s_auth_face_login_active.store(false);
    s_live_busy.store(false);
    set_face_overlay(false, nullptr, nullptr);
    s_last_recognition_us.store(0);
    s_last_pose_request_us.store(0);
    s_last_pc_pose_request_us.store(0);
    if (sw_live) {
        lv_obj_clear_state(sw_live, LV_STATE_CHECKED);
    }
    release_models_if_idle(false, true);

    s_pose_enabled = true;
    s_pose_busy.store(false);
    s_pc_pose_busy.store(false);
    s_last_pose_request_us.store(0);
    s_last_pc_pose_request_us.store(0);
    clear_pose_overlay();
    g_ui_screen.store(static_cast<int>(UiScreen::Pose));
    push_ui_msg("姿态页面就绪");

    if (bsp_display_lock(portMAX_DELAY)) {
        if (create_page_checked(&scr_pose, create_pose_page, "pose")) {
            lv_scr_load(scr_pose);
        } else {
            s_pending_screen = &scr_pose;
            s_pending_create_fn = create_pose_page;
            s_pending_page_name = "pose";
            s_pending_start_pose = true;
            start_page_preloader();
            push_ui_msg("正在准备实时训练...");
            bsp_display_unlock();
            return;
        }
        bsp_display_unlock();
    }

    if (init_camera_pipeline() != ESP_OK) {
        push_ui_msg("摄像头启动失败，请检查排线和传感器配置。");
    }
}

static void show_static_product_page(lv_obj_t **screen, create_product_page_fn_t create_fn, const char *log_name)
{
    ESP_LOGI(TAG, "%s", log_name);
    video_player_stop_current();
    training_pause_if_running();
    s_enroll_cancel.store(true);
    s_live_rec = false;
    s_auth_face_login_active.store(false);
    s_live_busy.store(false);
    s_pose_enabled = false;
    s_pose_busy.store(false);
    s_pc_pose_busy.store(false);
    s_body_roi_busy.store(false);
    s_last_body_roi_request_us.store(0);
    s_body_roi_reacquire_requested.store(false);
    set_face_overlay(false, nullptr, nullptr);
    clear_pose_overlay();
    s_last_recognition_us.store(0);
    s_last_pose_request_us.store(0);
    s_last_pc_pose_request_us.store(0);
    if (sw_live) {
        lv_obj_clear_state(sw_live, LV_STATE_CHECKED);
    }
    if (kb_name) {
        lv_obj_add_flag(kb_name, LV_OBJ_FLAG_HIDDEN);
        lv_keyboard_set_textarea(kb_name, nullptr);
    }
    if (kb_auth) {
        lv_obj_add_flag(kb_auth, LV_OBJ_FLAG_HIDDEN);
        lv_keyboard_set_textarea(kb_auth, nullptr);
    }
    if (kb_settings) {
        lv_obj_add_flag(kb_settings, LV_OBJ_FLAG_HIDDEN);
        lv_keyboard_set_textarea(kb_settings, nullptr);
    }
    release_models_if_idle(true, true);
    g_ui_screen.store(static_cast<int>(UiScreen::Menu));

    if (screen && bsp_display_lock(portMAX_DELAY)) {
        if (*screen) {
            lv_scr_load(*screen);
        } else {
            s_pending_screen = screen;
            s_pending_create_fn = create_fn;
            s_pending_page_name = log_name;
            s_pending_start_pose = false;
            start_page_preloader();
            push_ui_msg("正在准备页面...");
        }
        bsp_display_unlock();
    }
}

static void show_start_cb(lv_event_t *e)
{
    (void)e;
    show_static_product_page(&scr_start, create_start_page, "Show start page");
}

static void show_correction_cb(lv_event_t *e)
{
    (void)e;
    destroy_page_if_inactive(&scr_correction);
    show_static_product_page(&scr_correction, create_correction_page, "Show correction page");
}

static void show_courses_cb(lv_event_t *e)
{
    (void)e;
    destroy_page_if_inactive(&scr_courses);
    show_static_product_page(&scr_courses, create_courses_page, "Show courses page");
}

static void show_plan_cb(lv_event_t *e)
{
    (void)e;
    show_static_product_page(&scr_plan, create_plan_page, "Show plan page");
}

static void show_report_cb(lv_event_t *e)
{
    (void)e;
    destroy_page_if_inactive(&scr_report);
    show_static_product_page(&scr_report, create_report_page, "Show report page");
}

static void show_settings_cb(lv_event_t *e)
{
    (void)e;
    show_static_product_page(&scr_settings, create_settings_page, "Show settings page");
}

static void show_music_cb(lv_event_t *e)
{
    (void)e;
    if (!s_sdcard_mounted) {
        start_sdcard_retry_task();
    }
    if (s_sdcard_mounted && music_track_count() == 0) {
        music_load_playlist();
        if (scr_music) {
            lv_obj_delete(scr_music);
            scr_music = nullptr;
        }
    }
    show_static_product_page(&scr_music, create_music_page, "Show music page");
}

static void music_play_toggle_cb(lv_event_t *e)
{
    (void)e;
    music_send_cmd(MusicCmdType::Toggle);
}

static void music_stop_cb(lv_event_t *e)
{
    (void)e;
    music_send_cmd(MusicCmdType::Stop);
}

static void music_next_cb(lv_event_t *e)
{
    (void)e;
    music_send_cmd(MusicCmdType::Next);
}

static void music_prev_cb(lv_event_t *e)
{
    (void)e;
    music_send_cmd(MusicCmdType::Prev);
}

static void music_track_cb(lv_event_t *e)
{
    const intptr_t index = (intptr_t)lv_event_get_user_data(e);
    music_send_cmd(MusicCmdType::PlayIndex, (int)index);
}

static void music_volume_cb(lv_event_t *e)
{
    lv_obj_t *slider = (lv_obj_t *)lv_event_get_target(e);
    if (!slider) {
        return;
    }
    music_send_cmd(MusicCmdType::SetVolume, (int)lv_slider_get_value(slider));
}

static void music_rescan_cb(lv_event_t *e)
{
    (void)e;
    music_load_playlist();
    lv_obj_t *old = scr_music;
    scr_music = nullptr;
    create_page_checked(&scr_music, create_music_page, "music");
    if (scr_music) {
        lv_scr_load(scr_music);
    }
    if (old && old != scr_music) {
        lv_obj_delete(old);
    }
}

static void enroll_cb(lv_event_t *e)
{
    (void)e;
    push_ui_msg("已点击录入");
    if (!ta_name || !s_enroll_sem) {
        push_ui_msg("界面未就绪");
        return;
    }

    const char *name = lv_textarea_get_text(ta_name);
    if (!name || strlen(name) == 0) {
        push_ui_msg("请先输入姓名");
        return;
    }

    if (s_enroll_busy.exchange(true)) {
        push_ui_msg("录入正在进行");
        return;
    }
    s_enroll_cancel.store(false);

    if (init_camera_pipeline() != ESP_OK) {
        s_enroll_busy.store(false);
        push_ui_msg("摄像头启动失败，请检查排线和传感器配置。");
        return;
    }

    snprintf(s_enroll_name, sizeof(s_enroll_name), "%s", name);
    xSemaphoreGive(s_enroll_sem);
    notify_worker();
    push_ui_msg("录入已排队，请保持人脸在画面中。");
}

static void sw_live_cb(lv_event_t *e)
{
    push_ui_msg("已切换实时识别");
    lv_obj_t *sw = (lv_obj_t *)lv_event_get_target(e);
    const bool next_live = lv_obj_has_state(sw, LV_STATE_CHECKED);
    s_pose_enabled = false;
    s_pose_busy.store(false);
    s_pc_pose_busy.store(false);
    s_last_pose_request_us.store(0);
    s_last_pc_pose_request_us.store(0);
    clear_pose_overlay();
    if (next_live) {
        release_models_if_idle(true, false);
    }
    s_live_rec = next_live;
    s_live_busy.store(false);
    s_frame_ctr = 0;
    if (!s_live_rec) {
        set_face_overlay(false, nullptr, nullptr);
        s_last_recognition_us.store(0);
    }
    push_ui_msg(s_live_rec ? "实时识别已开启" : "实时识别已关闭");
}

static void delete_id_cb(lv_event_t *e)
{
    (void)e;
    if (!s_rec || !ta_del_id) {
        push_ui_msg("特征库未就绪");
        return;
    }

    const char *text = lv_textarea_get_text(ta_del_id);
    char *end = nullptr;
    long id = strtol(text ? text : "", &end, 10);
    if (!text || end == text || id <= 0 || id > 65535) {
        push_ui_msg("请输入要删除的特征ID");
        return;
    }

    if (!lock_models(pdMS_TO_TICKS(1000))) {
        push_ui_msg("模型忙，请稍后重试。");
        return;
    }

    esp_err_t err = s_rec->delete_feat((uint16_t)id);
    const int count = s_rec->get_num_feats();
    unlock_models();

    if (err == ESP_OK) {
        erase_face_name((uint16_t)id);
        s_db_count.store(count);
        lv_textarea_set_text(ta_del_id, "");
        push_ui_msg("已删除特征 id=%ld", id);
    } else {
        push_ui_msg("删除失败: 未找到 id=%ld", id);
    }
}

static void delete_last_cb(lv_event_t *e)
{
    (void)e;
    if (!s_rec) {
        push_ui_msg("特征库未就绪");
        return;
    }

    if (!lock_models(pdMS_TO_TICKS(1000))) {
        push_ui_msg("模型忙，请稍后重试。");
        return;
    }

    const uint16_t id = s_rec->get_last_feat_id();
    esp_err_t err = s_rec->delete_last_feat();
    const int count = s_rec->get_num_feats();
    unlock_models();

    if (err == ESP_OK) {
        erase_face_name(id);
        s_db_count.store(count);
        push_ui_msg("已删除最后特征 id=%u", (unsigned)id);
    } else {
        push_ui_msg("没有可删除的特征");
    }
}

static void clear_db_cb(lv_event_t *e)
{
    (void)e;
    if (!s_rec) {
        push_ui_msg("特征库未就绪");
        return;
    }

    if (!lock_models(pdMS_TO_TICKS(1000))) {
        push_ui_msg("模型忙，请稍后重试。");
        return;
    }

    esp_err_t err = s_rec->clear_all_feats();
    unlock_models();

    if (err == ESP_OK) {
        erase_all_face_names();
        s_db_count.store(0);
        set_face_overlay(false, nullptr, nullptr);
        s_last_recognition_us.store(0);
        push_ui_msg("特征库已清空");
    } else {
        push_ui_msg("清空特征库失败");
    }
}

static void name_textarea_cb(lv_event_t *e)
{
    lv_event_code_t code = lv_event_get_code(e);
    lv_obj_t *target = (lv_obj_t *)lv_event_get_target(e);

    if (code == LV_EVENT_FOCUSED || code == LV_EVENT_CLICKED) {
        if (kb_name && target != kb_name) {
            lv_keyboard_set_textarea(kb_name, target);
            lv_obj_clear_flag(kb_name, LV_OBJ_FLAG_HIDDEN);
            lv_obj_move_foreground(kb_name);
        }
    } else if (code == LV_EVENT_DEFOCUSED || code == LV_EVENT_READY || code == LV_EVENT_CANCEL) {
        if (kb_name) {
            lv_obj_add_flag(kb_name, LV_OBJ_FLAG_HIDDEN);
            lv_keyboard_set_textarea(kb_name, nullptr);
        }
    }
}

static void style_textarea(lv_obj_t *ta)
{
    if (!ta) {
        return;
    }
    lv_obj_set_style_text_font(ta, font_14(), LV_PART_MAIN);
    lv_obj_set_style_text_color(ta, lv_color_hex(0x16232d), LV_PART_MAIN);
    lv_obj_set_style_bg_color(ta, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_border_color(ta, lv_color_hex(0xd9e2e8), LV_PART_MAIN);
    lv_obj_set_style_border_width(ta, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(ta, 8, LV_PART_MAIN);
    lv_obj_set_style_pad_left(ta, 10, LV_PART_MAIN);
    lv_obj_set_style_pad_right(ta, 10, LV_PART_MAIN);
    lv_obj_set_style_pad_top(ta, 6, LV_PART_MAIN);
    lv_obj_set_style_pad_bottom(ta, 6, LV_PART_MAIN);
}

static void shared_keyboard_cb(lv_event_t *e)
{
    lv_event_code_t code = lv_event_get_code(e);
    lv_obj_t *target = (lv_obj_t *)lv_event_get_target(e);
    lv_obj_t *kb = (lv_obj_t *)lv_event_get_user_data(e);
    if (!kb) {
        return;
    }

    if ((code == LV_EVENT_FOCUSED || code == LV_EVENT_CLICKED) && target != kb) {
        lv_keyboard_set_textarea(kb, target);
        lv_obj_clear_flag(kb, LV_OBJ_FLAG_HIDDEN);
        lv_obj_move_foreground(kb);
    } else if (code == LV_EVENT_DEFOCUSED || code == LV_EVENT_READY || code == LV_EVENT_CANCEL) {
        lv_obj_add_flag(kb, LV_OBJ_FLAG_HIDDEN);
        lv_keyboard_set_textarea(kb, nullptr);
    }
}

static void profile_sync_edit_state(void)
{
    strlcpy(s_profile_gender_sel, s_user_profile.gender[0] ? s_user_profile.gender : "男", sizeof(s_profile_gender_sel));
    strlcpy(s_profile_goal_sel, s_user_profile.goal[0] ? s_user_profile.goal : "塑形", sizeof(s_profile_goal_sel));
    strlcpy(s_profile_focus_sel, s_user_profile.focus[0] ? s_user_profile.focus : "全身", sizeof(s_profile_focus_sel));
}

static void update_profile_choice_ui(void)
{
    if (!lbl_profile_choice) {
        return;
    }
    char line[128];
    snprintf(line,
             sizeof(line),
             "已选: %s / %s / %s",
             s_profile_gender_sel,
             s_profile_goal_sel,
             s_profile_focus_sel);
    lv_label_set_text(lbl_profile_choice, line);
}

static uint16_t parse_u16_or_default(const char *text, uint16_t fallback, uint16_t min_v, uint16_t max_v)
{
    char *end = nullptr;
    long v = strtol(text ? text : "", &end, 10);
    if (!text || end == text || v < min_v || v > max_v) {
        return fallback;
    }
    return (uint16_t)v;
}

static void destroy_page_if_inactive(lv_obj_t **screen)
{
    if (!screen || !*screen || *screen == lv_scr_act()) {
        return;
    }
    lv_obj_delete(*screen);
    *screen = nullptr;
    reset_page_refs(screen);
}

static void refresh_profile_dependent_pages(void)
{
    destroy_page_if_inactive(&scr_menu);
    destroy_page_if_inactive(&scr_start);
    destroy_page_if_inactive(&scr_plan);
    destroy_page_if_inactive(&scr_report);
    destroy_page_if_inactive(&scr_settings);
}

static void enter_menu_after_profile_save(void)
{
    g_ui_screen.store(static_cast<int>(UiScreen::Menu));
    if (!scr_menu) {
        create_page_checked(&scr_menu, create_menu, "overview");
    }
    if (scr_menu) {
        lv_scr_load_anim(scr_menu, LV_SCREEN_LOAD_ANIM_FADE_ON, 260, 0, false);
    } else if (lbl_profile_status) {
        lv_label_set_text(lbl_profile_status, "主页创建失败，稍后会重试");
    }
}

static void profile_choice_cb(lv_event_t *e)
{
    const char *data = (const char *)lv_event_get_user_data(e);
    if (!data) {
        return;
    }
    const char *sep = strchr(data, ':');
    if (!sep) {
        return;
    }
    const size_t key_len = (size_t)(sep - data);
    const char *value = sep + 1;
    if (strncmp(data, "gender", key_len) == 0) {
        strlcpy(s_profile_gender_sel, value, sizeof(s_profile_gender_sel));
    } else if (strncmp(data, "goal", key_len) == 0) {
        strlcpy(s_profile_goal_sel, value, sizeof(s_profile_goal_sel));
    } else if (strncmp(data, "focus", key_len) == 0) {
        strlcpy(s_profile_focus_sel, value, sizeof(s_profile_focus_sel));
    }
    update_profile_choice_ui();
}

static void profile_save_cb(lv_event_t *e)
{
    (void)e;
    user_profile_t next = {};
    user_profile_set_defaults(&next);
    const char *name = ta_profile_name ? lv_textarea_get_text(ta_profile_name) : "";
    if (!name || name[0] == '\0') {
        name = s_auth_user[0] ? s_auth_user : "用户";
    }
    strlcpy(next.name, name, sizeof(next.name));
    strlcpy(next.gender, s_profile_gender_sel, sizeof(next.gender));
    strlcpy(next.goal, s_profile_goal_sel, sizeof(next.goal));
    strlcpy(next.focus, s_profile_focus_sel, sizeof(next.focus));
    next.height_cm = parse_u16_or_default(ta_profile_height ? lv_textarea_get_text(ta_profile_height) : "",
                                          s_user_profile.height_cm ? s_user_profile.height_cm : 170,
                                          90,
                                          230);
    next.weight_kg = parse_u16_or_default(ta_profile_weight ? lv_textarea_get_text(ta_profile_weight) : "",
                                          s_user_profile.weight_kg ? s_user_profile.weight_kg : 65,
                                          25,
                                          220);
    next.minutes_per_day = parse_u16_or_default(ta_profile_minutes ? lv_textarea_get_text(ta_profile_minutes) : "",
                                                s_user_profile.minutes_per_day ? s_user_profile.minutes_per_day : 20,
                                                5,
                                                180);
    next.complete = true;

    esp_err_t err = save_user_profile(&next);
    if (err != ESP_OK) {
        if (lbl_profile_status) {
            char line[80];
            snprintf(line, sizeof(line), "保存失败: %s", esp_err_to_name(err));
            lv_label_set_text(lbl_profile_status, line);
        }
        return;
    }

    training_select_index(user_recommended_training_index());
    refresh_profile_dependent_pages();
    push_ui_msg("用户资料已保存");
    enter_menu_after_profile_save();
}

static void profile_default_cb(lv_event_t *e)
{
    (void)e;
    if (ta_profile_name) {
        lv_textarea_set_text(ta_profile_name, s_auth_user[0] ? s_auth_user : "用户");
    }
    if (ta_profile_height) {
        lv_textarea_set_text(ta_profile_height, "170");
    }
    if (ta_profile_weight) {
        lv_textarea_set_text(ta_profile_weight, "65");
    }
    if (ta_profile_minutes) {
        lv_textarea_set_text(ta_profile_minutes, "20");
    }
    strlcpy(s_profile_gender_sel, "男", sizeof(s_profile_gender_sel));
    strlcpy(s_profile_goal_sel, "塑形", sizeof(s_profile_goal_sel));
    strlcpy(s_profile_focus_sel, "全身", sizeof(s_profile_focus_sel));
    update_profile_choice_ui();
}

static void profile_edit_cb(lv_event_t *e)
{
    (void)e;
    profile_sync_edit_state();
    destroy_page_if_inactive(&scr_profile);
    if (!scr_profile) {
        create_page_checked(&scr_profile, create_profile_page, "profile");
    }
    if (scr_profile) {
        g_ui_screen.store(static_cast<int>(UiScreen::Profile));
        lv_scr_load_anim(scr_profile, LV_SCREEN_LOAD_ANIM_FADE_ON, 220, 0, false);
    }
}

static void set_auth_status(const char *text)
{
    if (lbl_auth_status) {
        lv_label_set_text(lbl_auth_status, text ? text : "");
    }
    if (text) {
        push_ui_msg("%s", text);
    }
}

static bool get_auth_inputs(char *user, size_t user_len, char *pass, size_t pass_len)
{
    const char *u = ta_auth_name ? lv_textarea_get_text(ta_auth_name) : "";
    const char *p = ta_auth_pass ? lv_textarea_get_text(ta_auth_pass) : "";
    strlcpy(user, u ? u : "", user_len);
    strlcpy(pass, p ? p : "", pass_len);
    return user[0] != '\0';
}

static void auth_password_login_cb(lv_event_t *e)
{
    (void)e;
    char user[32] = {};
    char pass[64] = {};
    if (!get_auth_inputs(user, sizeof(user), pass, sizeof(pass)) || pass[0] == '\0') {
        set_auth_status("请输入用户名和密码");
        return;
    }
    if (!s_auth_has_password.load()) {
        set_auth_status("还没有密码账号，请先注册");
        return;
    }
    if (strcmp(user, s_auth_saved_user) == 0 && strcmp(pass, s_auth_saved_pass) == 0) {
        strlcpy(s_auth_user, user, sizeof(s_auth_user));
        s_auth_unlock_pending.store(true);
        set_auth_status("密码验证通过");
    } else {
        set_auth_status("用户名或密码不正确");
    }
}

static void auth_password_register_cb(lv_event_t *e)
{
    (void)e;
    char user[32] = {};
    char pass[64] = {};
    if (!get_auth_inputs(user, sizeof(user), pass, sizeof(pass)) || pass[0] == '\0') {
        set_auth_status("注册需要用户名和密码");
        return;
    }
    esp_err_t err = save_auth_account(user, pass);
    if (err == ESP_OK) {
        strlcpy(s_auth_user, user, sizeof(s_auth_user));
        s_auth_unlock_pending.store(true);
        set_auth_status("注册成功");
    } else {
        char line[80];
        snprintf(line, sizeof(line), "注册失败: %s", esp_err_to_name(err));
        set_auth_status(line);
    }
}

static void auth_face_register_cb(lv_event_t *e)
{
    (void)e;
    char user[32] = {};
    char pass[64] = {};
    if (!get_auth_inputs(user, sizeof(user), pass, sizeof(pass))) {
        set_auth_status("人脸注册需要用户名");
        return;
    }
    if (pass[0] != '\0') {
        save_auth_account(user, pass);
    }
    if (s_enroll_busy.exchange(true)) {
        set_auth_status("人脸录入正在进行");
        return;
    }
    if (!s_enroll_sem) {
        s_enroll_busy.store(false);
        set_auth_status("人脸录入任务未就绪");
        return;
    }
    if (init_camera_pipeline() != ESP_OK) {
        s_enroll_busy.store(false);
        set_auth_status("摄像头启动失败");
        return;
    }
    if (s_auth_brand) {
        lv_obj_add_flag(s_auth_brand, LV_OBJ_FLAG_HIDDEN);
    }

    s_enroll_cancel.store(false);
    strlcpy(s_enroll_name, user, sizeof(s_enroll_name));
    xSemaphoreGive(s_enroll_sem);
    notify_worker();
    set_auth_status("请面向摄像头，正在录入人脸");
}

static void auth_face_login_cb(lv_event_t *e)
{
    (void)e;
    if (!lock_models(pdMS_TO_TICKS(1000))) {
        set_auth_status("模型正忙，请稍后再试");
        return;
    }
    release_pose_model();
    const bool face_ready = ensure_face_models();
    const int face_count = s_rec ? s_rec->get_num_feats() : 0;
    unlock_models();
    if (!face_ready) {
        set_auth_status("人脸模型加载失败");
        return;
    }
    if (face_count <= 0) {
        set_auth_status("还没有录入人脸，请先录入");
        return;
    }
    if (init_camera_pipeline() != ESP_OK) {
        set_auth_status("摄像头启动失败");
        return;
    }
    if (s_auth_brand) {
        lv_obj_add_flag(s_auth_brand, LV_OBJ_FLAG_HIDDEN);
    }
    s_pose_enabled = false;
    s_auth_face_login_active.store(true);
    s_live_rec = true;
    s_live_busy.store(false);
    s_last_recognition_us.store(0);
    g_ui_screen.store(static_cast<int>(UiScreen::Auth));
    set_auth_status("请面向摄像头进行人脸登录");
}

static bool start_wifi_reconnect_from_ui(const char *busy_msg)
{
    if (s_wifi_reconnect_task) {
        push_ui_msg("%s", busy_msg ? busy_msg : "Wi-Fi 正在连接，请稍候");
        return false;
    }

    BaseType_t ok = xTaskCreatePinnedToCore(wifi_reconnect_task,
                                            "wifi_reconnect",
                                            4096,
                                            nullptr,
                                            2,
                                            &s_wifi_reconnect_task,
                                            0);
    if (ok != pdPASS) {
        s_wifi_reconnect_task = nullptr;
        push_ui_msg("Wi-Fi 连接任务创建失败");
        return false;
    }
    return true;
}

static void wifi_connect_cb(lv_event_t *e)
{
    (void)e;
    const char *ssid = ta_wifi_ssid ? lv_textarea_get_text(ta_wifi_ssid) : "";
    const char *pass = ta_wifi_pass ? lv_textarea_get_text(ta_wifi_pass) : "";
    if (!ssid || ssid[0] == '\0') {
        push_ui_msg("请输入 Wi-Fi SSID");
        return;
    }
    strlcpy(s_wifi_ssid, ssid, sizeof(s_wifi_ssid));
    strlcpy(s_wifi_password, pass ? pass : "", sizeof(s_wifi_password));
    esp_err_t err = save_wifi_credentials(s_wifi_ssid, s_wifi_password);
    if (err != ESP_OK) {
        push_ui_msg("Wi-Fi 保存失败: %s", esp_err_to_name(err));
        return;
    }
    push_ui_msg("Wi-Fi 配置已保存");
    start_wifi_reconnect_from_ui("Wi-Fi 正在连接，请稍候");
}

static void wifi_reconnect_cb(lv_event_t *e)
{
    (void)e;
    if (s_wifi_ssid[0] == '\0') {
        push_ui_msg("没有可用的 Wi-Fi 配置");
        return;
    }
    start_wifi_reconnect_from_ui("Wi-Fi 正在重连，请稍候");
}

static void wifi_disconnect_cb(lv_event_t *e)
{
    (void)e;
    pc_pose_close_socket();
    s_pc_pose_wifi_ready.store(false);
    set_wifi_ip_text(nullptr);
    s_wifi_ui_state.store((int)WifiUiState::Disconnected);
    esp_wifi_disconnect();
    push_ui_msg("Wi-Fi 已断开");
}

static void wifi_restore_default_cb(lv_event_t *e)
{
    (void)e;
    esp_err_t err = erase_wifi_credentials();
    if (err != ESP_OK) {
        push_ui_msg("默认 Wi-Fi 恢复失败: %s", esp_err_to_name(err));
        return;
    }
    strlcpy(s_wifi_ssid, CONFIG_PC_POSE_WIFI_SSID, sizeof(s_wifi_ssid));
    strlcpy(s_wifi_password, CONFIG_PC_POSE_WIFI_PASSWORD, sizeof(s_wifi_password));
    if (ta_wifi_ssid) {
        lv_textarea_set_text(ta_wifi_ssid, s_wifi_ssid);
    }
    if (ta_wifi_pass) {
        lv_textarea_set_text(ta_wifi_pass, s_wifi_password);
    }
    push_ui_msg("已恢复默认热点配置");
    start_wifi_reconnect_from_ui("Wi-Fi 正在连接默认热点");
}

static void auth_clear_account_cb(lv_event_t *e)
{
    (void)e;
    esp_err_t err = erase_auth_account();
    if (err == ESP_OK) {
        if (ta_auth_name) {
            lv_textarea_set_text(ta_auth_name, "");
        }
        if (ta_auth_pass) {
            lv_textarea_set_text(ta_auth_pass, "");
        }
        set_auth_status("账号已清除，下次需要重新注册");
    } else {
        push_ui_msg("账号清除失败: %s", esp_err_to_name(err));
    }
}

static void auth_logout_cb(lv_event_t *e)
{
    (void)e;
    s_auth_user[0] = '\0';
    s_auth_face_login_active.store(false);
    s_auth_unlock_pending.store(false);
    s_live_rec = false;
    s_live_busy.store(false);
    set_face_overlay(false, nullptr, nullptr);
    if (!scr_auth) {
        create_page_checked(&scr_auth, create_auth_page, "auth");
    }
    if (scr_auth) {
        g_ui_screen.store(static_cast<int>(UiScreen::Auth));
        lv_scr_load_anim(scr_auth, LV_SCREEN_LOAD_ANIM_FADE_ON, 220, 0, false);
        push_ui_msg("已锁定，请重新登录");
    } else {
        push_ui_msg("登录页创建失败");
    }
}

static void face_clear_storage_cb(lv_event_t *e)
{
    (void)e;
    esp_err_t err = ESP_OK;
    if (lock_models(pdMS_TO_TICKS(1000))) {
        if (s_rec) {
            err = s_rec->clear_all_feats();
        }
        unlock_models();
    } else {
        err = ESP_ERR_TIMEOUT;
    }

    if (err == ESP_OK) {
        erase_all_face_names();
        char db_path[64];
        snprintf(db_path, sizeof(db_path), "%s/face.db", BSP_SPIFFS_MOUNT_POINT);
        if (!s_rec) {
            unlink(db_path);
        }
        s_db_count.store(0);
        set_face_overlay(false, nullptr, nullptr);
        s_last_recognition_us.store(0);
        push_ui_msg("人脸数据已清除");
    } else {
        push_ui_msg("人脸数据清除失败: %s", esp_err_to_name(err));
    }
}

static void create_auth_page(void)
{
    scr_auth = lv_obj_create(NULL);
    if (!scr_auth) {
        mark_lvgl_build_failed("auth screen");
        return;
    }
    style_screen(scr_auth);

    view_auth = make_preview(scr_auth);
    if (view_auth) {
        lv_obj_set_style_bg_color(view_auth, lv_color_hex(0xe9f2f0), LV_PART_MAIN);
    }
    if (view_auth) {
        cv_auth = lv_canvas_create(view_auth);
        if (!cv_auth) {
            mark_lvgl_build_failed("auth canvas");
        } else {
            lv_obj_clear_flag(cv_auth, LV_OBJ_FLAG_CLICKABLE);
            lv_obj_center(cv_auth);
        }
    }

    s_auth_brand = view_auth ? lv_obj_create(view_auth) : nullptr;
    if (s_auth_brand) {
        lv_obj_remove_style_all(s_auth_brand);
        lv_obj_set_size(s_auth_brand, 460, 210);
        lv_obj_center(s_auth_brand);
        lv_obj_clear_flag(s_auth_brand, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_t *logo = lv_image_create(s_auth_brand);
        if (logo) {
            lv_image_set_src(logo, &img_competition_logo);
            lv_obj_set_pos(logo, 0, 0);
        }
        make_text(s_auth_brand, "智能健身镜", 0, 78, 360, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
        make_text(s_auth_brand, "姿态纠正训练系统", 0, 112, 360, font_16(), 0x314551, LV_LABEL_LONG_DOT);
        make_text(s_auth_brand, "登录后进入训练、课程、计划和技术视图。", 0, 154, 430, font_14(), 0x687783);
    }

    lv_obj_t *panel = make_side_panel(scr_auth);
    if (!panel) {
        return;
    }
    lv_obj_set_style_bg_color(panel, lv_color_hex(0xf7f9fc), LV_PART_MAIN);

    make_text(panel, "用户登录", 0, 12, SIDE_PANEL_W - 24, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(panel, "支持密码登录和本地人脸登录", 0, 44, SIDE_PANEL_W - 24, font_14(), 0x687783, LV_LABEL_LONG_DOT);

    make_text(panel, "用户名", 0, 96, SIDE_PANEL_W - 24, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    ta_auth_name = lv_textarea_create(panel);
    lv_obj_set_size(ta_auth_name, SIDE_PANEL_W - 24, 42);
    lv_obj_set_pos(ta_auth_name, 0, 120);
    lv_textarea_set_one_line(ta_auth_name, true);
    lv_textarea_set_max_length(ta_auth_name, 31);
    lv_textarea_set_placeholder_text(ta_auth_name, "输入用户名");
    if (s_auth_saved_user[0]) {
        lv_textarea_set_text(ta_auth_name, s_auth_saved_user);
    }
    style_textarea(ta_auth_name);

    make_text(panel, "密码", 0, 176, SIDE_PANEL_W - 24, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    ta_auth_pass = lv_textarea_create(panel);
    lv_obj_set_size(ta_auth_pass, SIDE_PANEL_W - 24, 42);
    lv_obj_set_pos(ta_auth_pass, 0, 200);
    lv_textarea_set_one_line(ta_auth_pass, true);
    lv_textarea_set_password_mode(ta_auth_pass, true);
    lv_textarea_set_password_bullet(ta_auth_pass, "*");
    lv_textarea_set_max_length(ta_auth_pass, 63);
    lv_textarea_set_placeholder_text(ta_auth_pass, "输入密码");
    style_textarea(ta_auth_pass);

    const int32_t auth_content_w = SIDE_PANEL_W - 24;
    const int32_t auth_btn_gap = 8;
    const int32_t auth_btn_w = (auth_content_w - auth_btn_gap) / 2;
    const int32_t auth_btn_x2 = auth_btn_w + auth_btn_gap;
    make_product_button(panel, 0, 262, auth_btn_w, 42, "密码登录", auth_password_login_cb, true);
    make_product_button(panel, auth_btn_x2, 262, auth_btn_w, 42, "注册", auth_password_register_cb, false);
    make_product_button(panel, 0, 320, auth_btn_w, 42, "人脸登录", auth_face_login_cb, true);
    make_product_button(panel, auth_btn_x2, 320, auth_btn_w, 42, "录入人脸", auth_face_register_cb, false);

    lv_obj_t *hint = make_card(panel, 0, 392, auth_content_w, 92);
    make_text(hint, "提示", 0, 0, SIDE_PANEL_W - 52, font_14(), 0x145883, LV_LABEL_LONG_DOT);
    make_text(hint, "首次使用可以先注册密码账号，也可以输入用户名后录入人脸。", 0, 30, SIDE_PANEL_W - 52, font_14(), 0x687783);

    lbl_auth_status = lv_label_create(panel);
    lv_obj_set_width(lbl_auth_status, SIDE_PANEL_W - 24);
    lv_label_set_long_mode(lbl_auth_status, LV_LABEL_LONG_WRAP);
    lv_label_set_text(lbl_auth_status, s_auth_has_password.load() ? "请输入密码或使用人脸登录" : "请先注册账号");
    style_text(lbl_auth_status, font_14(), 0x314551);
    lv_obj_set_style_bg_color(lbl_auth_status, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(lbl_auth_status, LV_OPA_80, LV_PART_MAIN);
    lv_obj_set_style_pad_all(lbl_auth_status, 10, LV_PART_MAIN);
    lv_obj_set_style_radius(lbl_auth_status, 8, LV_PART_MAIN);
    lv_obj_align(lbl_auth_status, LV_ALIGN_BOTTOM_MID, 0, 0);

    kb_auth = lv_keyboard_create(scr_auth);
    lv_obj_set_size(kb_auth, SCREEN_W, 230);
    lv_obj_align(kb_auth, LV_ALIGN_BOTTOM_MID, 0, 0);
    lv_obj_add_flag(kb_auth, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_event_cb(ta_auth_name, shared_keyboard_cb, LV_EVENT_ALL, kb_auth);
    lv_obj_add_event_cb(ta_auth_pass, shared_keyboard_cb, LV_EVENT_ALL, kb_auth);
    lv_obj_add_event_cb(kb_auth, shared_keyboard_cb, LV_EVENT_READY, kb_auth);
    lv_obj_add_event_cb(kb_auth, shared_keyboard_cb, LV_EVENT_CANCEL, kb_auth);
}

static void create_profile_page(void)
{
    scr_profile = lv_obj_create(NULL);
    if (!scr_profile) {
        mark_lvgl_build_failed("profile screen");
        return;
    }
    style_product_screen(scr_profile);
    profile_sync_edit_state();

    lv_obj_t *intro = make_card(scr_profile, 52, 54, 336, 492);
    if (intro) {
        lv_obj_set_style_bg_color(intro, lv_color_hex(0xfafdfc), LV_PART_MAIN);
        lv_obj_t *logo = lv_image_create(intro);
        if (logo) {
            lv_image_set_src(logo, &img_competition_logo);
            lv_obj_set_pos(logo, 0, 2);
        }
        make_text(intro, "完善基础信息", 0, 96, 270, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
        make_text(intro,
                  "系统会根据你的目标和重点，推荐合适训练，并把训练结果保存到本地记录。",
                  0,
                  136,
                  286,
                  font_14(),
                  0x687783,
                  LV_LABEL_LONG_WRAP);
        make_pill(intro, 0, 220, 128, "资料保存", 0x22a06b);
        make_pill(intro, 144, 220, 128, "训练推荐", 0x145883);
        make_pill(intro, 0, 268, 128, "本地记录", 0xd9822b);
        make_pill(intro, 144, 268, 128, "后续分析", 0x2f7dd1);
        make_text(intro, "登录账号", 0, 350, 100, font_14(), 0x687783, LV_LABEL_LONG_DOT);
        make_text(intro, s_auth_user[0] ? s_auth_user : "未命名用户", 0, 382, 250, font_18(), 0x145883, LV_LABEL_LONG_DOT);
    }

    lv_obj_t *form = make_card(scr_profile, 420, 54, 552, 492);
    if (!form) {
        return;
    }
    make_text(form, "你的训练资料", 0, 0, 220, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(form, "以后可以在设备设置里修改。", 244, 4, 260, font_14(), 0x687783, LV_LABEL_LONG_DOT);

    make_text(form, "昵称", 0, 54, 54, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    ta_profile_name = lv_textarea_create(form);
    if (ta_profile_name) {
        lv_obj_set_size(ta_profile_name, 150, 38);
        lv_obj_set_pos(ta_profile_name, 56, 44);
        lv_textarea_set_one_line(ta_profile_name, true);
        lv_textarea_set_max_length(ta_profile_name, 31);
        lv_textarea_set_placeholder_text(ta_profile_name, "昵称");
        lv_textarea_set_text(ta_profile_name,
                             s_user_profile.complete ? s_user_profile.name : (s_auth_user[0] ? s_auth_user : ""));
        style_textarea(ta_profile_name);
    }

    make_text(form, "身高", 226, 54, 54, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    ta_profile_height = lv_textarea_create(form);
    if (ta_profile_height) {
        char text[12];
        snprintf(text, sizeof(text), "%u", (unsigned)(s_user_profile.height_cm ? s_user_profile.height_cm : 170));
        lv_obj_set_size(ta_profile_height, 72, 38);
        lv_obj_set_pos(ta_profile_height, 280, 44);
        lv_textarea_set_one_line(ta_profile_height, true);
        lv_textarea_set_max_length(ta_profile_height, 3);
        lv_textarea_set_text(ta_profile_height, text);
        style_textarea(ta_profile_height);
    }

    make_text(form, "体重", 370, 54, 54, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    ta_profile_weight = lv_textarea_create(form);
    if (ta_profile_weight) {
        char text[12];
        snprintf(text, sizeof(text), "%u", (unsigned)(s_user_profile.weight_kg ? s_user_profile.weight_kg : 65));
        lv_obj_set_size(ta_profile_weight, 72, 38);
        lv_obj_set_pos(ta_profile_weight, 424, 44);
        lv_textarea_set_one_line(ta_profile_weight, true);
        lv_textarea_set_max_length(ta_profile_weight, 3);
        lv_textarea_set_text(ta_profile_weight, text);
        style_textarea(ta_profile_weight);
    }

    make_text(form, "性别", 0, 112, 90, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_profile_choice_button(form, 76, 104, 74, "男", "gender:男");
    make_profile_choice_button(form, 160, 104, 74, "女", "gender:女");

    make_text(form, "目标", 0, 176, 90, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_profile_choice_button(form, 76, 168, 76, "塑形", "goal:塑形");
    make_profile_choice_button(form, 162, 168, 76, "减脂", "goal:减脂");
    make_profile_choice_button(form, 248, 168, 76, "力量", "goal:力量");
    make_profile_choice_button(form, 334, 168, 76, "康复", "goal:康复");

    make_text(form, "重点", 0, 240, 90, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_profile_choice_button(form, 76, 232, 76, "全身", "focus:全身");
    make_profile_choice_button(form, 162, 232, 76, "下肢", "focus:下肢");
    make_profile_choice_button(form, 248, 232, 76, "核心", "focus:核心");
    make_profile_choice_button(form, 334, 232, 76, "上肢", "focus:上肢");

    make_text(form, "每日时间", 0, 304, 90, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    ta_profile_minutes = lv_textarea_create(form);
    if (ta_profile_minutes) {
        char text[12];
        snprintf(text, sizeof(text), "%u", (unsigned)(s_user_profile.minutes_per_day ? s_user_profile.minutes_per_day : 20));
        lv_obj_set_size(ta_profile_minutes, 82, 38);
        lv_obj_set_pos(ta_profile_minutes, 86, 294);
        lv_textarea_set_one_line(ta_profile_minutes, true);
        lv_textarea_set_max_length(ta_profile_minutes, 3);
        lv_textarea_set_text(ta_profile_minutes, text);
        style_textarea(ta_profile_minutes);
    }
    make_text(form, "分钟", 178, 304, 54, font_14(), 0x687783, LV_LABEL_LONG_DOT);

    lbl_profile_choice = make_text(form, "", 0, 354, 500, font_14(), 0x145883, LV_LABEL_LONG_DOT);
    lbl_profile_status = make_text(form, "保存后会进入主页，并更新推荐训练。", 0, 382, 500, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_product_button(form, 0, 424, 120, 38, "保存进入", profile_save_cb, true);
    make_product_button(form, 136, 424, 116, 38, "使用默认", profile_default_cb, false);
    update_profile_choice_ui();

    kb_profile = lv_keyboard_create(scr_profile);
    if (kb_profile) {
        lv_obj_set_size(kb_profile, SCREEN_W, 230);
        lv_obj_align(kb_profile, LV_ALIGN_BOTTOM_MID, 0, 0);
        lv_obj_add_flag(kb_profile, LV_OBJ_FLAG_HIDDEN);
        if (ta_profile_name) {
            lv_obj_add_event_cb(ta_profile_name, shared_keyboard_cb, LV_EVENT_ALL, kb_profile);
        }
        if (ta_profile_height) {
            lv_obj_add_event_cb(ta_profile_height, shared_keyboard_cb, LV_EVENT_ALL, kb_profile);
        }
        if (ta_profile_weight) {
            lv_obj_add_event_cb(ta_profile_weight, shared_keyboard_cb, LV_EVENT_ALL, kb_profile);
        }
        if (ta_profile_minutes) {
            lv_obj_add_event_cb(ta_profile_minutes, shared_keyboard_cb, LV_EVENT_ALL, kb_profile);
        }
        lv_obj_add_event_cb(kb_profile, shared_keyboard_cb, LV_EVENT_READY, kb_profile);
        lv_obj_add_event_cb(kb_profile, shared_keyboard_cb, LV_EVENT_CANCEL, kb_profile);
    }
}

static void create_menu(void)
{
    scr_menu = lv_obj_create(NULL);
    if (!scr_menu) {
        mark_lvgl_build_failed("overview screen");
        return;
    }
    lv_obj_t *main = make_product_shell(scr_menu,
                                        "overview",
                                        "今日概览",
                                        "训练、纠正、课程和设备状态集中入口。");
    if (!main) {
        return;
    }
    add_top_status(main, "Wi-Fi", "PC姿态", "本地待机");

    const int rec_index = user_recommended_training_index();
    const training_profile_t *rec_profile = training_profile_by_index(rec_index);
    char hero_desc[220];
    if (s_user_profile.complete) {
        snprintf(hero_desc,
                 sizeof(hero_desc),
                 "%s的当前目标是%s，重点关注%s。建议先完成%s，再进入实时纠正。",
                 s_user_profile.name,
                 s_user_profile.goal,
                 s_user_profile.focus,
                 rec_profile->name);
    } else {
        snprintf(hero_desc, sizeof(hero_desc), "完善基础信息后，系统会根据目标自动推荐训练。");
    }

    lv_obj_t *hero = make_card(main, 24, 94, 500, 168);
    make_text(hero, "今日推荐", 0, 0, 180, font_14(), 0x145883, LV_LABEL_LONG_DOT);
    make_text(hero, rec_profile->name, 0, 30, 250, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(hero, hero_desc, 0, 62, 292, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
    make_product_button(hero, 0, 116, 118, 38, "选择训练", show_start_cb, true);
    make_product_button(hero, 130, 116, 84, 38, "计划", show_plan_cb, false);
    make_product_button(hero, 226, 116, 84, 38, "资料", profile_edit_cb, false);

    if (hero) {
        lv_obj_t *mirror = lv_obj_create(hero);
        if (mirror) {
            lv_obj_set_pos(mirror, 328, 8);
            lv_obj_set_size(mirror, 142, 138);
            lv_obj_clear_flag(mirror, LV_OBJ_FLAG_SCROLLABLE);
            lv_obj_set_style_bg_color(mirror, lv_color_hex(0x0b344f), LV_PART_MAIN);
            lv_obj_set_style_border_width(mirror, 0, LV_PART_MAIN);
            lv_obj_set_style_radius(mirror, 8, LV_PART_MAIN);
            make_text(mirror, "实时镜像", 16, 14, 110, font_14(), 0xffffff, LV_LABEL_LONG_DOT);
            make_progress(mirror, 16, 110, 110, 76, 0x22a06b);
            make_dot(mirror, 62, 44, 0x22a06b);
            make_dot(mirror, 48, 68, 0x22a06b);
            make_dot(mirror, 78, 68, 0x22a06b);
            make_dot(mirror, 50, 94, 0x22a06b);
            make_dot(mirror, 76, 94, 0x22a06b);
        } else {
            mark_lvgl_build_failed("overview mirror");
        }
    }

    char sessions_text[24];
    char score_text[24];
    char count_text[24];
    char elapsed_text[24];
    snprintf(sessions_text, sizeof(sessions_text), "%u次", (unsigned)s_training_record.sessions);
    snprintf(score_text,
             sizeof(score_text),
             s_training_record.sessions > 0 ? "%d分" : "--",
             s_training_record.last_score);
    snprintf(count_text,
             sizeof(count_text),
             s_training_record.sessions > 0 ? "%d/%d" : "--",
             s_training_record.last_count,
             s_training_record.last_target);
    format_mmss(s_training_record.last_elapsed_ms, elapsed_text, sizeof(elapsed_text));
    make_metric(main, 544, 94, 116, 78, "训练", sessions_text, s_training_record.sessions > 0 ? 76 : 12, 0x145883);
    make_metric(main, 676, 94, 116, 78, "得分", score_text, s_training_record.last_score, 0x22a06b);
    make_metric(main, 544, 184, 116, 78, "次数", count_text, s_training_record.sessions > 0 ? 82 : 10, 0xd9822b);
    make_metric(main, 676, 184, 116, 78, "用时", s_training_record.sessions > 0 ? elapsed_text : "--", 72, 0x2f7dd1);

    const char *feature_titles[] = {"训练", "纠正", "课程", "设置"};
    const char *feature_marks[] = {"训", "纠", "课", "设"};
    const char *feature_desc[] = {
        "先选动作，再进入实时训练。",
        "查看角度、稳定性和常见错误。",
        "按目标浏览训练内容。",
        "管理Wi-Fi、PC服务和显示。",
    };
    lv_event_cb_t feature_cb[] = {show_start_cb, show_correction_cb, show_courses_cb, show_settings_cb};
    const uint32_t feature_color[] = {0x145883, 0x22a06b, 0xd9822b, 0x6f65c8};
    for (int i = 0; i < 4; i++) {
        const int32_t x = 24 + i * 192;
        lv_obj_t *card = make_card(main, x, 284, 176, 122);
        if (!card) {
            continue;
        }
        lv_obj_add_flag(card, LV_OBJ_FLAG_CLICKABLE);
        lv_obj_add_event_cb(card, feature_cb[i], LV_EVENT_CLICKED, NULL);
        lv_obj_t *mark = lv_obj_create(card);
        if (!mark) {
            mark_lvgl_build_failed("feature mark");
            continue;
        }
        lv_obj_remove_style_all(mark);
        lv_obj_set_size(mark, 28, 28);
        lv_obj_set_style_radius(mark, 8, LV_PART_MAIN);
        lv_obj_set_style_bg_color(mark, lv_color_hex(feature_color[i]), LV_PART_MAIN);
        lv_obj_set_style_bg_opa(mark, LV_OPA_COVER, LV_PART_MAIN);
        lv_obj_t *mark_label = make_text(mark, feature_marks[i], 0, 3, 28, font_14(), 0xffffff, LV_LABEL_LONG_CLIP);
        if (mark_label) {
            lv_obj_set_style_text_align(mark_label, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
        }
        make_text(card, feature_titles[i], 38, 2, 112, font_14(), 0x16232d, LV_LABEL_LONG_DOT);
        make_text(card, feature_desc[i], 0, 44, 148, font_14(), 0x687783);
    }

    lv_obj_t *schedule = make_card(main, 24, 426, 376, 134);
    make_text(schedule, "今日任务", 0, 0, 180, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    char task_line[96];
    char rec_target[48];
    training_target_text_by_index(rec_index, rec_target, sizeof(rec_target));
    snprintf(task_line, sizeof(task_line), "%s       %s", rec_profile->name, rec_target);
    make_text(schedule, task_line, 0, 36, 330, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_text(schedule, "动作纠正       实时评分", 0, 62, 330, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_text(schedule, "训练报告       自动保存", 0, 88, 330, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_text(schedule, "音量", 206, 88, 40, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    slider_menu_volume = lv_slider_create(schedule);
    if (slider_menu_volume) {
        lv_obj_set_pos(slider_menu_volume, 250, 94);
        lv_obj_set_size(slider_menu_volume, 86, 10);
        lv_slider_set_range(slider_menu_volume, 0, 100);
        lv_slider_set_value(slider_menu_volume, music_get_state_copy().volume, LV_ANIM_OFF);
        lv_obj_set_style_bg_color(slider_menu_volume, lv_color_hex(0xe4edf1), LV_PART_MAIN);
        lv_obj_set_style_bg_color(slider_menu_volume, lv_color_hex(0x145883), LV_PART_INDICATOR);
        lv_obj_set_style_bg_color(slider_menu_volume, lv_color_hex(0x145883), LV_PART_KNOB);
        lv_obj_add_event_cb(slider_menu_volume, music_volume_cb, LV_EVENT_VALUE_CHANGED, NULL);
    } else {
        mark_lvgl_build_failed("overview volume");
    }

    lv_obj_t *health = make_card(main, 416, 426, 376, 134);
    make_text(health, "设备状态", 0, 0, 180, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_menu_weather = make_text(health, "天气: 待接入", 218, 4, 122, font_14(), 0x145883, LV_LABEL_LONG_DOT);
    lbl_menu_wifi = make_text(health, "Wi-Fi: --", 0, 32, 330, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    lbl_menu_pc = make_text(health, "PC服务: --", 0, 56, 330, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    lbl_menu_ip = make_text(health, "IP: --", 0, 80, 330, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    lbl_menu_datetime = make_text(health, "时间: 等待校时", 0, 104, 330, font_14(), 0x145883, LV_LABEL_LONG_DOT);
    update_wifi_labels_ui();
    update_datetime_ui();
}

static void create_camera_page(void)
{
    scr_cam = lv_obj_create(NULL);
    if (!scr_cam) {
        mark_lvgl_build_failed("camera screen");
        return;
    }
    style_screen(scr_cam);

    view_cam = make_preview(scr_cam);
    cv_cam = lv_canvas_create(view_cam);
    if (cv_cam) {
        lv_obj_clear_flag(cv_cam, LV_OBJ_FLAG_CLICKABLE);
        lv_obj_center(cv_cam);
    } else {
        mark_lvgl_build_failed("camera canvas");
    }

    lv_obj_t *panel = make_side_panel(scr_cam);

    lv_obj_t *title = lv_label_create(panel);
    if (title) {
        set_label(title, "摄像头");
        lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 4);
    } else {
        mark_lvgl_build_failed("camera title");
    }

    lv_obj_t *back = make_button(panel, SIDE_PANEL_W - 24, 52, "返回");
    if (back) {
        lv_obj_align(back, LV_ALIGN_TOP_MID, 0, 48);
        lv_obj_add_event_cb(back, show_menu_cb, LV_EVENT_CLICKED, NULL);
    }

    lv_obj_t *start = make_button(panel, SIDE_PANEL_W - 24, 52, "选择训练");
    if (start) {
        lv_obj_align(start, LV_ALIGN_TOP_MID, 0, 116);
        lv_obj_add_event_cb(start, show_start_cb, LV_EVENT_CLICKED, NULL);
    }

    lv_obj_t *pose = make_button(panel, SIDE_PANEL_W - 24, 52, "姿态模式");
    if (pose) {
        lv_obj_align(pose, LV_ALIGN_TOP_MID, 0, 184);
        lv_obj_add_event_cb(pose, show_pose_cb, LV_EVENT_CLICKED, NULL);
    }
}

static void create_pose_page(void)
{
    scr_pose = lv_obj_create(NULL);
    if (!scr_pose) {
        mark_lvgl_build_failed("pose screen");
        return;
    }
    style_screen(scr_pose);

    view_pose = make_preview(scr_pose);
    cv_pose = lv_canvas_create(view_pose);
    if (cv_pose) {
        lv_obj_clear_flag(cv_pose, LV_OBJ_FLAG_CLICKABLE);
        lv_obj_center(cv_pose);
    } else {
        mark_lvgl_build_failed("pose canvas");
    }

    lv_obj_t *panel = make_side_panel(scr_pose);
    lv_obj_set_style_bg_color(panel, lv_color_hex(0xf7f9fc), LV_PART_MAIN);

    make_text(panel, "实时训练", 0, 8, SIDE_PANEL_W - 24, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_train_subtitle = make_text(panel, "", 0, 38, SIDE_PANEL_W - 24, font_14(), 0x687783, LV_LABEL_LONG_DOT);

    lv_obj_t *back = make_product_button(panel, 0, 68, SIDE_PANEL_W - 24, 38, "返回", show_menu_cb, false);
    (void)back;
    make_product_button(panel, 0, 114, SIDE_PANEL_W - 24, 38, "选择项目", show_start_cb, true);

    lv_obj_t *primary = make_product_button(panel, 0, 154, 96, 36, "开始", training_toggle_cb, true);
    if (primary) {
        btn_train_primary_label = lv_obj_get_child(primary, 0);
    }
    make_product_button(panel, 108, 154, 92, 36, "结束", training_finish_cb, false);

    lv_obj_t *count_card = make_card(panel, 0, 202, SIDE_PANEL_W - 24, 58);
    if (count_card) {
        lv_obj_set_style_shadow_width(count_card, 0, LV_PART_MAIN);
        make_text(count_card, "当前次数", 0, 0, 100, font_14(), 0x687783, LV_LABEL_LONG_DOT);
        lbl_train_count = make_text(count_card, "0 / 0", 0, 24, 150, font_18(), 0x145883, LV_LABEL_LONG_DOT);
    }

    lv_obj_t *time_card = make_card(panel, 0, 270, 94, 58);
    if (time_card) {
        lv_obj_set_style_shadow_width(time_card, 0, LV_PART_MAIN);
        make_text(time_card, "用时", 0, 0, 60, font_14(), 0x687783, LV_LABEL_LONG_DOT);
        lbl_train_timer = make_text(time_card, "00:00", 0, 24, 80, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    }

    lv_obj_t *score_card = make_card(panel, 106, 270, 94, 58);
    if (score_card) {
        lv_obj_set_style_shadow_width(score_card, 0, LV_PART_MAIN);
        make_text(score_card, "得分", 0, 0, 60, font_14(), 0x687783, LV_LABEL_LONG_DOT);
        lbl_train_score = make_text(score_card, "0", 0, 24, 70, font_18(), 0x22a06b, LV_LABEL_LONG_DOT);
    }

    lv_obj_t *cue = make_card(panel, 0, 338, SIDE_PANEL_W - 24, 72);
    if (cue) {
        lv_obj_set_style_bg_color(cue, lv_color_hex(0xeaf7f1), LV_PART_MAIN);
        lv_obj_set_style_shadow_width(cue, 0, LV_PART_MAIN);
        make_text(cue, "提示", 0, 0, 60, font_14(), 0x1a7f53, LV_LABEL_LONG_DOT);
        lbl_train_cue = make_text(cue,
                                  "选择训练项目后开始",
                                  0,
                                  24,
                                  SIDE_PANEL_W - 52,
                                  font_14(),
                                  0x314551,
                                  LV_LABEL_LONG_DOT);
    }

    lv_obj_t *state_card = make_card(panel, 0, 420, SIDE_PANEL_W - 24, 54);
    if (state_card) {
        lv_obj_set_style_shadow_width(state_card, 0, LV_PART_MAIN);
        lv_obj_set_style_bg_color(state_card, lv_color_hex(0xffffff), LV_PART_MAIN);
        make_text(state_card, "状态", 0, 0, 50, font_14(), 0x687783, LV_LABEL_LONG_DOT);
        lbl_train_state = make_text(state_card,
                                    "准备  第1组\n点0  PC 0.0FPS",
                                    54,
                                    0,
                                    SIDE_PANEL_W - 86,
                                    font_14(),
                                    0x314551,
                                    LV_LABEL_LONG_DOT);
        if (lbl_train_state) {
            lv_obj_set_height(lbl_train_state, 40);
            lv_obj_set_style_text_line_space(lbl_train_state, 2, LV_PART_MAIN);
        }
    }

    const int32_t record_y = 486;
    const int32_t music_y = 518;
    const int32_t status_y = 550;

    s_record_button = make_product_button(panel, 0, record_y, 88, 28, "录制15秒", record_toggle_cb, false);
    s_record_status_label = make_text(panel, "录制: 待命", 100, record_y + 6, SIDE_PANEL_W - 124, font_14(), 0x687783, LV_LABEL_LONG_DOT);

    make_product_button(panel, 0, music_y, 88, 28, "停音乐", music_stop_cb, false);
    lbl_pose_music = make_text(panel, "音乐: --", 100, music_y + 6, SIDE_PANEL_W - 124, font_14(), 0x687783, LV_LABEL_LONG_DOT);

    lbl_pose_status = lv_label_create(panel);
    if (!lbl_pose_status) {
        mark_lvgl_build_failed("pose status");
        return;
    }
    lv_obj_set_width(lbl_pose_status, SIDE_PANEL_W - 24);
    lv_obj_set_height(lbl_pose_status, 30);
    lv_label_set_long_mode(lbl_pose_status, LV_LABEL_LONG_DOT);
    set_label(lbl_pose_status, "准备就绪");
    lv_obj_set_style_text_font(lbl_pose_status, font_14(), LV_PART_MAIN);
    lv_obj_set_style_text_color(lbl_pose_status, lv_color_hex(0x263244), LV_PART_MAIN);
    lv_obj_set_style_pad_all(lbl_pose_status, 4, LV_PART_MAIN);
    lv_obj_set_style_bg_color(lbl_pose_status, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(lbl_pose_status, LV_OPA_80, LV_PART_MAIN);
    lv_obj_set_style_radius(lbl_pose_status, 8, LV_PART_MAIN);
    lv_obj_set_pos(lbl_pose_status, 0, status_y);
    update_training_ui();
    update_music_ui();
}

static void create_start_page(void)
{
    scr_start = lv_obj_create(NULL);
    if (!scr_start) {
        mark_lvgl_build_failed("start screen");
        return;
    }
    lv_obj_t *main = make_product_shell(scr_start,
                                        "start",
                                        "选择训练",
                                        "先选择一个动作，再进入清爽的实时训练界面。");
    if (!main) {
        return;
    }
    add_top_status(main, "摄像头", "姿态纠正", "自由模式");

    for (int i = 0; i < TRAINING_PROFILE_COUNT; i++) {
        const training_profile_t *profile = training_profile_by_index(i);
        const int32_t x = 24 + (i % 4) * 192;
        const int32_t y = 94 + (i / 4) * 124;
        lv_obj_t *card = make_card(main, x, y, 176, 108);
        if (!card) {
            continue;
        }
        lv_obj_add_flag(card, LV_OBJ_FLAG_CLICKABLE);
        lv_obj_add_event_cb(card, training_card_cb, LV_EVENT_CLICKED, (void *)(intptr_t)i);
        make_dot(card, 2, 4, profile->color);
        make_text(card, profile->name, 20, 0, 134, font_16(), 0x16232d, LV_LABEL_LONG_DOT);
        make_text(card, profile->meta, 0, 34, 138, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
        make_text(card, "点击选择", 0, 74, 120, font_14(), 0x145883, LV_LABEL_LONG_DOT);
    }

    lv_obj_t *config = make_card(main, 24, 360, 768, 176);
    make_text(config, "训练配置", 0, 0, 200, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_start_current = make_text(config, "当前: 深蹲", 0, 38, 166, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    lbl_start_target = make_text(config, "目标: 4组 x 12次", 174, 38, 176, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    lbl_start_focus = make_text(config, "重点: 膝盖轨迹、深度、躯干前倾", 358, 38, 372, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_product_button(config, 0, 74, 108, 36, "开始实时", training_start_selected_cb, true);
    make_product_button(config, 120, 74, 82, 36, "课程", show_courses_cb, false);
    make_product_button(config, 214, 74, 72, 36, "计划", show_plan_cb, false);
    make_text(config, "组数", 310, 82, 38, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_product_button(config, 350, 76, 30, 30, "-", training_goal_adjust_cb, false, (void *)(intptr_t)-1);
    lbl_start_sets_value = make_text(config, "4", 388, 82, 24, font_14(), 0x16232d, LV_LABEL_LONG_DOT);
    make_product_button(config, 414, 76, 30, 30, "+", training_goal_adjust_cb, false, (void *)(intptr_t)1);
    lbl_start_reps_title = make_text(config, "每组次数", 462, 82, 70, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_product_button(config, 536, 76, 30, 30, "-", training_goal_adjust_cb, false, (void *)(intptr_t)-2);
    lbl_start_reps_value = make_text(config, "12", 574, 82, 34, font_14(), 0x16232d, LV_LABEL_LONG_DOT);
    make_product_button(config, 612, 76, 30, 30, "+", training_goal_adjust_cb, false, (void *)(intptr_t)2);
    lbl_start_guide = make_text(config,
                                 "指导: 侧前方45°拍全身；髋膝踝都要入镜；下蹲到髋部明显下降再站起，膝盖跟脚尖同向。",
                                 0,
                                 120,
                                 724,
                                 font_14(),
                                 0x4e6472,
                                 LV_LABEL_LONG_WRAP);
    if (lbl_start_guide) {
        lv_obj_set_height(lbl_start_guide, 44);
    }
    update_training_ui();
}

static std::string offline_result_path_for(const video_item_t &item)
{
    std::string out = OFFLINE_RESULT_DIR;
    if (!out.empty() && out.back() != '/') {
        out.push_back('/');
    }
    out += item.id.empty() ? "result" : item.id;
    out += ".txt";
    return out;
}

static std::string offline_result_video_path_for(const video_item_t &item)
{
    std::string out = OFFLINE_RESULT_DIR;
    if (!out.empty() && out.back() != '/') {
        out.push_back('/');
    }
    out += item.id.empty() ? "result" : item.id;
    out += "_result.mjpeg";
    return out;
}

static video_item_t offline_playback_item_for(const video_item_t &item)
{
    video_item_t out = item;
    const std::string result_video = offline_result_video_path_for(item);
    if (access(result_video.c_str(), F_OK) == 0) {
        out.path = result_video;
    }
    return out;
}

static const char *offline_tip_for_action(const std::string &action)
{
    if (action == "squat") {
        return "深蹲重点：髋部后坐、膝盖跟脚尖同向、起身完全站直。";
    }
    if (action == "pushup") {
        return "俯卧撑重点：肩髋踝保持一线，胸口下降后再推起。";
    }
    if (action == "bench_press") {
        return "卧推重点：两侧同步下放和推起，手腕保持稳定。";
    }
    if (action == "pullup") {
        return "引体向上重点：先到底部伸展，再向上拉到下巴接近横杆。";
    }
    if (action == "deadlift") {
        return "硬拉重点：背部稳定，髋膝协调伸展，顶部锁定。";
    }
    if (action == "dumbbell_curl") {
        return "哑铃弯举重点：上臂尽量稳定，完整屈肘并控制下放。";
    }
    if (action == "situp") {
        return "仰卧起坐重点：腹部发力卷起，不要猛拉颈部。";
    }
    return "检测重点：保持身体完整入镜，动作节奏稳定。";
}

typedef struct {
    int score;
    int depth;
    int knee;
    int hip;
    int torso;
    int balance;
    int track;
    const char *rule;
    const char *correction;
    const char *view;
} offline_eval_hint_t;

static offline_eval_hint_t offline_eval_hint_for_action(const std::string &action)
{
    if (action == "squat") {
        return {86,
                84,
                78,
                82,
                80,
                82,
                78,
                "髋-膝-踝判断膝角，肩-髋-膝判断髋角和躯干，完整下蹲并回到站立后计一次。",
                "下蹲深度、膝盖轨迹、躯干前倾和左右发力会进入评分。",
                "建议正前方或45度侧前方，保证脚踝和膝盖都在画面内。"};
    }
    if (action == "pushup") {
        return {82,
                78,
                76,
                84,
                80,
                78,
                76,
                "肩-肘-腕判断下降幅度，肩-髋-踝判断身体线条，下降到位并推起伸展后计一次。",
                "胸口下降不足、塌腰、抬臀和左右不同步会触发纠错。",
                "建议侧面或45度侧前方，肩髋脚踝需要横向入镜。"};
    }
    if (action == "bench_press") {
        return {82,
                78,
                78,
                80,
                80,
                78,
                76,
                "肩-肘-腕判断下放和推起，左右手臂同步性进入评分。",
                "下放幅度不足、两侧不同步和手腕不稳会触发纠错。",
                "建议侧前方拍摄，让两侧手臂和肩部都清楚入镜。"};
    }
    if (action == "pullup") {
        return {80,
                76,
                78,
                76,
                74,
                76,
                74,
                "手腕高于肩部作为握杆参考，头部相对手腕上移后再回到底部伸展计一次。",
                "高度不足、身体摆动、手臂行程不完整会触发纠错。",
                "建议正面或侧面，头、肩、手腕完整入镜。"};
    }
    if (action == "deadlift") {
        return {82,
                82,
                78,
                82,
                78,
                80,
                76,
                "肩髋膝踝估计髋部折叠和顶部锁定，底部发力并站直后计一次。",
                "背部不稳、髋膝节奏不协调和顶部未锁定会触发纠错。",
                "建议侧前方拍摄，肩髋膝踝要完整入镜。"};
    }
    if (action == "plank") {
        return {84,
                86,
                80,
                86,
                82,
                80,
                76,
                "肩-髋-踝接近一条线后开始计时，完全塌腰或离开支撑姿态会暂停。",
                "塌腰、抬臀、核心不稳会触发纠错，轻微波动不会立刻中断。",
                "建议侧面拍摄，让肩、髋、踝在同一画面内。"};
    }
    if (action == "dumbbell_curl" || action == "curl") {
        return {82,
                78,
                76,
                78,
                80,
                76,
                74,
                "肩-肘-腕判断屈肘角度，顶部屈肘后控制下放到底部伸展计一次。",
                "幅度不足、身体借力和左右不同步会触发纠错。",
                "建议正面或45度侧前方，上半身和双臂完整入镜。"};
    }
    if (action == "situp") {
        return {82,
                78,
                76,
                80,
                76,
                78,
                74,
                "肩-髋-膝判断躯干卷起角度，卷起后控制回到底部计一次。",
                "卷起幅度不足、颈部发力和下放失控会触发纠错。",
                "建议侧面拍摄，肩、髋、膝和头部尽量完整入镜。"};
    }
    return {78,
            76,
            76,
            76,
            76,
            76,
            72,
            "按照当前动作的关键点角度、幅度和稳定性进行评分。",
            "动作不完整或关键点不稳定时给出纠错提示。",
            "建议让全身完整入镜并保持稳定光照。"};
}

static bool offline_load_result(const video_item_t &item, std::string *out)
{
    if (!out) {
        return false;
    }
    const std::string path = offline_result_path_for(item);
    FILE *fp = fopen(path.c_str(), "r");
    if (!fp) {
        return false;
    }
    char buf[256];
    out->clear();
    while (fgets(buf, sizeof(buf), fp)) {
        *out += buf;
    }
    fclose(fp);
    return !out->empty();
}

static std::string offline_result_value(const std::string &raw, const char *key)
{
    if (!key) {
        return "";
    }
    std::string prefix = key;
    prefix += "=";
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t end = raw.find('\n', pos);
        if (end == std::string::npos) {
            end = raw.size();
        }
        if (raw.compare(pos, prefix.size(), prefix) == 0) {
            return raw.substr(pos + prefix.size(), end - pos - prefix.size());
        }
        pos = end + 1;
    }
    return "";
}

static std::string offline_format_result_summary(const std::string &raw)
{
    const std::string profile = offline_result_value(raw, "profile");
    if (profile.empty()) {
        return raw;
    }
    const std::string count = offline_result_value(raw, "count");
    const std::string score = offline_result_value(raw, "score");
    const std::string rep_score = offline_result_value(raw, "rep_score");
    const std::string sample_fps = offline_result_value(raw, "sample_fps");
    const std::string sample_target = offline_result_value(raw, "sample_target");
    const std::string frame_count = offline_result_value(raw, "frame_count");
    const std::string elapsed_ms = offline_result_value(raw, "elapsed_ms");
    const std::string depth = offline_result_value(raw, "score_depth");
    const std::string knee = offline_result_value(raw, "score_knee");
    const std::string hip = offline_result_value(raw, "score_hip");
    const std::string torso = offline_result_value(raw, "score_torso");
    const std::string balance = offline_result_value(raw, "score_balance");
    const std::string track = offline_result_value(raw, "score_track");
    const std::string correction = offline_result_value(raw, "correction");
    const std::string view = offline_result_value(raw, "view");
    std::string rep_log = offline_result_value(raw, "rep_log");
    for (char &ch : rep_log) {
        if (ch == '|') {
            ch = '\n';
        }
    }

    std::string out;
    out.reserve(760);
    out += "动作：";
    out += profile;
    out += "  采样：";
    out += sample_fps.empty() ? "-" : sample_fps;
    out += "FPS / ";
    out += sample_target.empty() ? "-" : sample_target;
    out += "帧  视频帧：";
    out += frame_count.empty() ? "-" : frame_count;
    out += "  耗时：";
    out += elapsed_ms.empty() ? "-" : elapsed_ms;
    out += "ms\n次数：";
    out += count.empty() ? "-" : count;
    out += "  总分：";
    out += score.empty() ? "-" : score;
    out += "  动作均分:";
    out += rep_score.empty() ? "-" : rep_score;
    out += "\n分项：幅度";
    out += depth.empty() ? "-" : depth;
    out += " 膝";
    out += knee.empty() ? "-" : knee;
    out += " 髋";
    out += hip.empty() ? "-" : hip;
    out += " 躯干";
    out += torso.empty() ? "-" : torso;
    out += " 稳定";
    out += balance.empty() ? "-" : balance;
    out += " 跟踪";
    out += track.empty() ? "-" : track;
    out += "\n纠错：";
    out += correction.empty() ? "保持动作完整和稳定。" : correction;
    out += "\n视角：";
    out += view.empty() ? "尽量让全身完整入镜。" : view;
    if (!rep_log.empty()) {
        out += "\n\n单次动作：\n";
        out += rep_log;
    }
    return out;
}

typedef struct {
    size_t start;
    size_t len;
} offline_jpeg_frame_t;

typedef struct {
    int frame_count = 0;
    int sample_count = 0;
    int valid_frames = 0;
    int count = 0;
    int score = 0;
    int depth = 0;
    int knee = 0;
    int hip = 0;
    int torso = 0;
    int balance = 0;
    int track = 0;
    int rep_score = 0;
    int elapsed_ms = 0;
    std::string result_video;
    std::string status;
    std::string rep_log;
} offline_analysis_result_t;

typedef struct {
    float signal = 0.0f;
    int depth = 0;
    int knee = 0;
    int hip = 0;
    int torso = 0;
    int balance = 0;
    int track = 0;
    int total = 0;
} offline_frame_eval_t;

static int offline_score_from_unit(float v)
{
    return std::max(0, std::min(100, (int)lroundf(training_clampf(v, 0.0f, 1.0f) * 100.0f)));
}

static int offline_weighted_score(const offline_frame_eval_t &e,
                                  float depth_w,
                                  float knee_w,
                                  float hip_w,
                                  float torso_w,
                                  float balance_w,
                                  float track_w)
{
    return std::max(0,
                    std::min(100,
                             (int)lroundf((float)e.depth * depth_w +
                                          (float)e.knee * knee_w +
                                          (float)e.hip * hip_w +
                                          (float)e.torso * torso_w +
                                          (float)e.balance * balance_w +
                                          (float)e.track * track_w)));
}

static int offline_pose_valid_count(const pose_result_t &pose, float *avg_score)
{
    int valid = 0;
    float sum = 0.0f;
    const int n = std::min<int>(POSE_KPTS, pose.keypoints.size());
    for (int i = 0; i < n; ++i) {
        const float score = pose.keypoints[i].score;
        if (score >= 0.08f) {
            valid++;
            sum += score;
        }
    }
    if (avg_score) {
        *avg_score = valid > 0 ? sum / (float)valid : 0.0f;
    }
    return valid;
}

static bool offline_best_pose(const std::vector<pose_result_t> &results, pose_result_t *out, int *valid_count, float *avg_score)
{
    if (valid_count) {
        *valid_count = 0;
    }
    if (avg_score) {
        *avg_score = 0.0f;
    }
    if (!out || results.empty()) {
        return false;
    }

    int best_index = -1;
    float best_quality = -1.0f;
    int best_valid = 0;
    float best_avg = 0.0f;
    for (int i = 0; i < (int)results.size(); ++i) {
        float avg = 0.0f;
        const int valid = offline_pose_valid_count(results[i], &avg);
        const float area = (float)std::max(0, results[i].x2 - results[i].x1) *
                           (float)std::max(0, results[i].y2 - results[i].y1);
        const float quality = avg * 100.0f + valid * 3.0f + area * 0.00002f;
        if (quality > best_quality) {
            best_quality = quality;
            best_index = i;
            best_valid = valid;
            best_avg = avg;
        }
    }
    if (best_index < 0 || best_valid < 3) {
        return false;
    }
    *out = results[best_index];
    if (valid_count) {
        *valid_count = best_valid;
    }
    if (avg_score) {
        *avg_score = best_avg;
    }
    return true;
}

typedef struct {
    bool ok = false;
    bool left_ok = false;
    bool right_ok = false;
    float elbow_angle = 180.0f;
    float balance_delta = 0.0f;
    float confidence = 0.0f;
    pose_keypoint_t shoulder = {};
    pose_keypoint_t elbow = {};
    pose_keypoint_t wrist = {};
} offline_arm_metrics_t;

static bool offline_arm_metrics(const pose_result_t &pose, float min_score, offline_arm_metrics_t *out)
{
    if (!out) {
        return false;
    }
    *out = {};
    float angle_sum = 0.0f;
    float weight_sum = 0.0f;
    float left_angle = 0.0f;
    float right_angle = 0.0f;
    float best_weight = -1.0f;
    pose_keypoint_t best_shoulder = {};
    pose_keypoint_t best_elbow = {};
    pose_keypoint_t best_wrist = {};

    for (int side = 0; side < 2; ++side) {
        const bool left = side == 0;
        pose_keypoint_t shoulder = {};
        pose_keypoint_t elbow = {};
        pose_keypoint_t wrist = {};
        if (!training_get_keypoint(pose, left ? POSE_KPT_LEFT_SHOULDER : POSE_KPT_RIGHT_SHOULDER, min_score, &shoulder) ||
            !training_get_keypoint(pose, left ? POSE_KPT_LEFT_ELBOW : POSE_KPT_RIGHT_ELBOW, min_score, &elbow) ||
            !training_get_keypoint(pose, left ? POSE_KPT_LEFT_WRIST : POSE_KPT_RIGHT_WRIST, min_score, &wrist)) {
            continue;
        }
        const float angle = training_angle_deg(shoulder, elbow, wrist);
        const float weight = std::max(0.05f, (shoulder.score + elbow.score + wrist.score) / 3.0f);
        angle_sum += angle * weight;
        weight_sum += weight;
        if (left) {
            out->left_ok = true;
            left_angle = angle;
        } else {
            out->right_ok = true;
            right_angle = angle;
        }
        if (weight > best_weight) {
            best_weight = weight;
            best_shoulder = shoulder;
            best_elbow = elbow;
            best_wrist = wrist;
        }
    }

    if (weight_sum <= 0.0f) {
        return false;
    }
    out->ok = true;
    out->elbow_angle = angle_sum / weight_sum;
    out->balance_delta = (out->left_ok && out->right_ok) ? fabsf(left_angle - right_angle) : 0.0f;
    out->confidence = training_clampf(weight_sum / 1.2f, 0.0f, 1.0f);
    out->shoulder = best_shoulder;
    out->elbow = best_elbow;
    out->wrist = best_wrist;
    return true;
}

typedef struct {
    bool ok = false;
    float line_angle = 175.0f;
    float torso_lean = 0.0f;
    pose_keypoint_t shoulder = {};
    pose_keypoint_t hip = {};
    pose_keypoint_t ankle = {};
} offline_body_line_t;

static bool offline_body_line_metrics(const pose_result_t &pose, float min_score, offline_body_line_t *out)
{
    if (!out) {
        return false;
    }
    *out = {};
    pose_keypoint_t shoulder = {};
    pose_keypoint_t hip = {};
    pose_keypoint_t ankle = {};
    if (training_avg_keypoint(pose, POSE_KPT_LEFT_SHOULDER, POSE_KPT_RIGHT_SHOULDER, min_score, &shoulder) &&
        training_avg_keypoint(pose, POSE_KPT_LEFT_HIP, POSE_KPT_RIGHT_HIP, min_score, &hip) &&
        training_avg_keypoint(pose, POSE_KPT_LEFT_ANKLE, POSE_KPT_RIGHT_ANKLE, min_score * 0.75f, &ankle)) {
        out->ok = true;
    } else {
        float best_score = -1.0f;
        for (int side = 0; side < 2; ++side) {
            const bool left = side == 0;
            pose_keypoint_t s = {};
            pose_keypoint_t h = {};
            pose_keypoint_t a = {};
            if (!training_get_keypoint(pose, left ? POSE_KPT_LEFT_SHOULDER : POSE_KPT_RIGHT_SHOULDER, min_score, &s) ||
                !training_get_keypoint(pose, left ? POSE_KPT_LEFT_HIP : POSE_KPT_RIGHT_HIP, min_score, &h) ||
                !training_get_keypoint(pose, left ? POSE_KPT_LEFT_ANKLE : POSE_KPT_RIGHT_ANKLE, min_score * 0.75f, &a)) {
                continue;
            }
            const float score = s.score + h.score + a.score;
            if (score > best_score) {
                best_score = score;
                shoulder = s;
                hip = h;
                ankle = a;
                out->ok = true;
            }
        }
    }
    if (!out->ok) {
        return false;
    }
    out->shoulder = shoulder;
    out->hip = hip;
    out->ankle = ankle;
    out->line_angle = training_angle_deg(shoulder, hip, ankle);
    const float torso_dx = fabsf((float)shoulder.x - (float)hip.x);
    const float torso_dy = fabsf((float)shoulder.y - (float)hip.y);
    out->torso_lean = atan2f(torso_dx, std::max(1.0f, torso_dy)) * 57.2957795f;
    return true;
}

static bool offline_squat_side_metrics(const pose_result_t &pose, bool left, squat_metrics_t *out)
{
    if (!out) {
        return false;
    }
    *out = {};

    const int shoulder_idx = left ? POSE_KPT_LEFT_SHOULDER : POSE_KPT_RIGHT_SHOULDER;
    const int alt_shoulder_idx = left ? POSE_KPT_RIGHT_SHOULDER : POSE_KPT_LEFT_SHOULDER;
    const int hip_idx = left ? POSE_KPT_LEFT_HIP : POSE_KPT_RIGHT_HIP;
    const int knee_idx = left ? POSE_KPT_LEFT_KNEE : POSE_KPT_RIGHT_KNEE;
    const int ankle_idx = left ? POSE_KPT_LEFT_ANKLE : POSE_KPT_RIGHT_ANKLE;

    pose_keypoint_t shoulder = {};
    pose_keypoint_t hip = {};
    pose_keypoint_t knee = {};
    pose_keypoint_t ankle = {};
    if (!training_get_keypoint(pose, shoulder_idx, 0.075f, &shoulder)) {
        if (!training_get_keypoint(pose, alt_shoulder_idx, 0.075f, &shoulder)) {
            return false;
        }
    }
    if (!training_get_keypoint(pose, hip_idx, 0.055f, &hip) ||
        !training_get_keypoint(pose, knee_idx, 0.055f, &knee)) {
        return false;
    }
    const bool ankle_ok = training_get_keypoint(pose, ankle_idx, 0.045f, &ankle);

    out->valid = true;
    out->confidence = ankle_ok ? (shoulder.score + hip.score + knee.score + ankle.score) * 0.25f :
                                 (shoulder.score + hip.score + knee.score) * 0.3333333f;
    out->knee_angle_valid = ankle_ok;
    out->knee_angle = ankle_ok ? training_angle_deg(hip, knee, ankle) : 180.0f;
    if (ankle_ok) {
        const float leg_dx = (float)knee.x - (float)ankle.x;
        const float leg_dy = (float)knee.y - (float)ankle.y;
        const float leg_len = std::max(32.0f, sqrtf(leg_dx * leg_dx + leg_dy * leg_dy));
        out->knee_track_delta = fabsf((float)knee.x - (float)ankle.x) / leg_len;
        out->knee_track_valid = true;
    }
    out->hip_angle = training_angle_deg(shoulder, hip, knee);
    const float torso_dx = fabsf((float)shoulder.x - (float)hip.x);
    const float torso_dy = fabsf((float)shoulder.y - (float)hip.y);
    out->torso_lean = atan2f(torso_dx, std::max(1.0f, torso_dy)) * 57.2957795f;
    const float denom = (float)knee.y - (float)shoulder.y;
    out->depth = denom > 18.0f ? training_clampf(((float)hip.y - (float)shoulder.y) / denom, -0.25f, 1.35f) : -1.0f;
    return out->depth >= -0.10f;
}

static bool offline_squat_metrics(const pose_result_t &pose, squat_metrics_t *out)
{
    if (!out) {
        return false;
    }
    *out = {};
    squat_metrics_t left = {};
    squat_metrics_t right = {};
    const bool left_ok = offline_squat_side_metrics(pose, true, &left);
    const bool right_ok = offline_squat_side_metrics(pose, false, &right);
    if (!left_ok && !right_ok) {
        return false;
    }
    if (left_ok && !right_ok) {
        *out = left;
        out->symmetry_delta = 0.0f;
        return true;
    }
    if (!left_ok && right_ok) {
        *out = right;
        out->symmetry_delta = 0.0f;
        return true;
    }

    const float lw = std::max(0.05f, left.confidence);
    const float rw = std::max(0.05f, right.confidence);
    const float sum = lw + rw;
    out->valid = true;
    out->both_sides = true;
    out->confidence = (left.confidence * lw + right.confidence * rw) / sum;

    float knee_sum = 0.0f;
    float knee_w = 0.0f;
    if (left.knee_angle_valid) {
        knee_sum += left.knee_angle * lw;
        knee_w += lw;
    }
    if (right.knee_angle_valid) {
        knee_sum += right.knee_angle * rw;
        knee_w += rw;
    }
    out->knee_angle_valid = knee_w > 0.0f;
    out->knee_angle = out->knee_angle_valid ? knee_sum / knee_w : 180.0f;

    float track_sum = 0.0f;
    float track_w = 0.0f;
    if (left.knee_track_valid) {
        track_sum += left.knee_track_delta * lw;
        track_w += lw;
    }
    if (right.knee_track_valid) {
        track_sum += right.knee_track_delta * rw;
        track_w += rw;
    }
    out->knee_track_valid = track_w > 0.0f;
    out->knee_track_delta = out->knee_track_valid ? track_sum / track_w : 0.0f;
    out->hip_angle = (left.hip_angle * lw + right.hip_angle * rw) / sum;
    out->torso_lean = (left.torso_lean * lw + right.torso_lean * rw) / sum;
    out->depth = (left.depth * lw + right.depth * rw) / sum;
    out->symmetry_delta = (left.knee_angle_valid && right.knee_angle_valid) ? fabsf(left.knee_angle - right.knee_angle) : 0.0f;
    return true;
}

static bool offline_eval_action_frame(const video_item_t &item,
                                      const pose_result_t &pose,
                                      int valid_count,
                                      float avg_score,
                                      offline_frame_eval_t *eval)
{
    if (!eval) {
        return false;
    }
    *eval = {};
    eval->track = training_track_score(valid_count, avg_score);

    const std::string &action = item.action;
    if (action == "deadlift") {
        squat_metrics_t m = {};
        if (!offline_squat_metrics(pose, &m)) {
            return false;
        }
        const float hip_fold = training_clampf((168.0f - m.hip_angle) / 62.0f, 0.0f, 1.0f);
        const float torso_fold = training_clampf((m.torso_lean - 10.0f) / 48.0f, 0.0f, 1.0f);
        eval->signal = training_clampf(hip_fold * 0.76f + torso_fold * 0.24f, 0.0f, 1.0f);
        eval->depth = offline_score_from_unit(eval->signal);
        eval->knee = m.knee_angle_valid ? training_range_score(m.knee_angle, 112.0f, 170.0f, 40.0f) : 68;
        eval->hip = training_range_score(m.hip_angle, 88.0f, 138.0f, 50.0f);
        eval->torso = offline_score_from_unit(1.0f - (m.torso_lean - 30.0f) / 30.0f);
        eval->balance = m.both_sides ? offline_score_from_unit(1.0f - (m.symmetry_delta - 14.0f) / 34.0f) : 76;
        eval->total = offline_weighted_score(*eval, 0.22f, 0.16f, 0.24f, 0.20f, 0.08f, 0.10f);
        return true;
    }

    if (action == "squat") {
        squat_metrics_t m = {};
        if (!offline_squat_metrics(pose, &m)) {
            return false;
        }
        float depth_unit = training_clampf((m.depth - 0.52f) / 0.18f, 0.0f, 1.0f);
        float depth_angle_unit = training_clampf((158.0f - m.hip_angle) / 58.0f, 0.0f, 1.0f);
        if (m.knee_angle_valid) {
            depth_angle_unit = std::max(depth_angle_unit, training_clampf((142.0f - m.knee_angle) / 42.0f, 0.0f, 1.0f));
        }
        eval->signal = training_clampf(std::max(depth_unit, depth_angle_unit), 0.0f, 1.0f);
        eval->depth = offline_score_from_unit(eval->signal);
        const int knee_angle_score = m.knee_angle_valid ?
            training_score_from01(1.0f - std::max(0.0f, std::max(82.0f - m.knee_angle, m.knee_angle - 142.0f)) / 44.0f) :
            68;
        const int knee_track_score = m.knee_track_valid ?
            training_score_from01(1.0f - (m.knee_track_delta - 0.20f) / 0.30f) :
            72;
        eval->knee = training_clamp_score((int)lroundf((float)knee_track_score * 0.62f + (float)knee_angle_score * 0.38f));
        eval->hip = offline_score_from_unit(1.0f - fabsf(m.hip_angle - 98.0f) / 68.0f);
        eval->torso = offline_score_from_unit(1.0f - (m.torso_lean - 26.0f) / 26.0f);
        eval->balance = m.both_sides ? offline_score_from_unit(1.0f - (m.symmetry_delta - 14.0f) / 30.0f) : 76;
        eval->total = offline_weighted_score(*eval, 0.30f, 0.20f, 0.15f, 0.15f, 0.10f, 0.10f);
        return true;
    }

    if (action == "pushup" || action == "bench_press" || action == "dumbbell_curl" || action == "curl") {
        offline_arm_metrics_t arm = {};
        if (!offline_arm_metrics(pose, 0.08f, &arm)) {
            return false;
        }
        const float elbow = arm.elbow_angle;
        const float balance = arm.balance_delta;
        const float confidence = arm.confidence;
        const bool is_curl = action == "dumbbell_curl" || action == "curl";
        const bool is_bench = action == "bench_press";
        offline_body_line_t body = {};
        const bool body_ok = offline_body_line_metrics(pose, 0.08f, &body);
        const float elbow_signal = training_clampf((176.0f - elbow) / 116.0f, 0.0f, 1.0f);
        float body_drop_signal = 0.0f;
        if (!is_curl && !is_bench) {
            const float arm_len = sqrtf((float)(arm.shoulder.x - arm.wrist.x) * (float)(arm.shoulder.x - arm.wrist.x) +
                                        (float)(arm.shoulder.y - arm.wrist.y) * (float)(arm.shoulder.y - arm.wrist.y));
            const float vertical_gap = (float)arm.wrist.y - (float)arm.shoulder.y;
            body_drop_signal = training_clampf((arm_len * 1.06f - vertical_gap) / std::max(28.0f, arm_len * 0.62f), 0.0f, 1.0f);
        }
        eval->signal = is_curl || is_bench ? elbow_signal :
            training_clampf(elbow_signal * 0.64f + body_drop_signal * 0.36f, 0.0f, 1.0f);
        eval->depth = is_curl ?
            training_range_score(elbow, 58.0f, 108.0f, 52.0f) :
            training_range_score(elbow, 72.0f, is_bench ? 112.0f : 108.0f, 48.0f);
        eval->knee = balance > 0.0f ? offline_score_from_unit(1.0f - (balance - 12.0f) / 40.0f) : 76;
        eval->hip = is_curl ? offline_score_from_unit(confidence) :
            (is_bench ? 82 : (body_ok ? training_range_score(body.line_angle, 154.0f, 180.0f, 34.0f) : 62));
        eval->torso = is_curl ? 82 :
            (is_bench ? 82 : (body_ok ? training_range_score(body.line_angle, 154.0f, 180.0f, 34.0f) : 62));
        eval->balance = balance > 0.0f ? offline_score_from_unit(1.0f - (balance - 16.0f) / 46.0f) : 78;
        if (is_curl) {
            eval->total = offline_weighted_score(*eval, 0.46f, 0.22f, 0.00f, 0.00f, 0.14f, 0.18f);
        } else {
            eval->total = offline_weighted_score(*eval, 0.34f, 0.18f, 0.20f, 0.10f, 0.08f, 0.10f);
        }
        return true;
    }

    if (action == "pullup") {
        offline_arm_metrics_t arm = {};
        pose_keypoint_t nose = {};
        pose_keypoint_t hip = {};
        if (!offline_arm_metrics(pose, 0.08f, &arm)) {
            return false;
        }
        const bool has_nose = training_get_keypoint(pose, 0, 0.08f, &nose);
        const bool has_hip = training_avg_keypoint(pose, POSE_KPT_LEFT_HIP, POSE_KPT_RIGHT_HIP, 0.08f, &hip);
        const float elbow = arm.elbow_angle;
        const float balance = arm.balance_delta;
        const float shoulder_to_wrist = std::max(44.0f, fabsf((float)arm.wrist.y - (float)arm.shoulder.y));
        const float height_gap = has_nose ? ((float)nose.y - (float)arm.wrist.y) :
                                 ((float)arm.shoulder.y - (float)arm.wrist.y);
        const float pull_height = training_clampf((shoulder_to_wrist * 0.70f - height_gap) /
                                                  std::max(30.0f, shoulder_to_wrist * 0.58f),
                                                  0.0f,
                                                  1.0f);
        const float elbow_pull = training_clampf((170.0f - elbow) / 86.0f, 0.0f, 1.0f);
        eval->signal = training_clampf(pull_height * 0.58f + elbow_pull * 0.42f, 0.0f, 1.0f);
        const float body_sway = has_hip ? fabsf((float)hip.x - (float)arm.shoulder.x) /
                                          std::max(80.0f, fabsf((float)hip.y - (float)arm.shoulder.y)) :
                                          0.22f;
        const int overhead_score = arm.wrist.y < arm.shoulder.y + 64 ? 100 : 70;
        eval->depth = training_score_from01((eval->signal - 0.46f) / 0.34f);
        eval->knee = training_range_score(elbow, 72.0f, 138.0f, 56.0f);
        eval->hip = offline_score_from_unit(1.0f - (body_sway - 0.18f) / 0.36f);
        eval->torso = eval->hip;
        eval->balance = balance > 0.0f ? offline_score_from_unit(1.0f - (balance - 12.0f) / 40.0f) : 76;
        eval->track = std::min(eval->track, overhead_score);
        eval->total = offline_weighted_score(*eval, 0.34f, 0.22f, 0.16f, 0.04f, 0.10f, 0.14f);
        return true;
    }

    if (action == "situp") {
        pose_keypoint_t shoulder = {};
        pose_keypoint_t hip = {};
        pose_keypoint_t knee = {};
        pose_keypoint_t nose = {};
        if (!training_avg_keypoint(pose, POSE_KPT_LEFT_SHOULDER, POSE_KPT_RIGHT_SHOULDER, 0.07f, &shoulder) ||
            !training_avg_keypoint(pose, POSE_KPT_LEFT_HIP, POSE_KPT_RIGHT_HIP, 0.07f, &hip) ||
            !training_avg_keypoint(pose, POSE_KPT_LEFT_KNEE, POSE_KPT_RIGHT_KNEE, 0.06f, &knee)) {
            return false;
        }
        const bool has_nose = training_get_keypoint(pose, 0, 0.06f, &nose);
        const float angle = training_angle_deg(shoulder, hip, knee);
        eval->signal = training_clampf((170.0f - angle) / 92.0f, 0.0f, 1.0f);
        eval->depth = training_range_score(angle, 82.0f, 126.0f, 64.0f);
        eval->knee = 76;
        eval->hip = offline_score_from_unit(1.0f - fabsf(angle - 96.0f) / 72.0f);
        eval->torso = eval->depth;
        eval->balance = has_nose ? offline_score_from_unit(1.0f - (fabsf((float)nose.x - (float)shoulder.x) - 130.0f) / 190.0f) : 76;
        eval->total = offline_weighted_score(*eval, 0.44f, 0.00f, 0.16f, 0.12f, 0.10f, 0.18f);
        return true;
    }

    if (action == "plank") {
        offline_body_line_t body = {};
        if (!offline_body_line_metrics(pose, 0.07f, &body)) {
            return false;
        }
        const float line_angle = body.line_angle;
        const float torso = body.torso_lean;
        eval->signal = training_clampf(1.0f - fabsf(line_angle - 180.0f) / 50.0f, 0.0f, 1.0f);
        eval->depth = training_range_score(line_angle, 154.0f, 180.0f, 34.0f);
        eval->knee = 80;
        eval->hip = eval->depth;
        eval->torso = training_score_from01((torso - 30.0f) / 42.0f);
        eval->balance = eval->depth;
        eval->total = offline_weighted_score(*eval, 0.34f, 0.00f, 0.12f, 0.24f, 0.08f, 0.22f);
        return true;
    }

    return false;
}

static int offline_weighted_score_for_action(const std::string &action, const offline_frame_eval_t &e)
{
    if (action == "squat") {
        return offline_weighted_score(e, 0.30f, 0.20f, 0.15f, 0.15f, 0.10f, 0.10f);
    }
    if (action == "deadlift") {
        return offline_weighted_score(e, 0.22f, 0.16f, 0.24f, 0.20f, 0.08f, 0.10f);
    }
    if (action == "pushup" || action == "bench_press") {
        return offline_weighted_score(e, 0.34f, 0.18f, 0.20f, 0.10f, 0.08f, 0.10f);
    }
    if (action == "pullup") {
        return offline_weighted_score(e, 0.34f, 0.22f, 0.16f, 0.04f, 0.10f, 0.14f);
    }
    if (action == "dumbbell_curl" || action == "curl") {
        return offline_weighted_score(e, 0.46f, 0.22f, 0.00f, 0.00f, 0.14f, 0.18f);
    }
    if (action == "situp") {
        return offline_weighted_score(e, 0.44f, 0.00f, 0.16f, 0.12f, 0.10f, 0.18f);
    }
    if (action == "plank") {
        return offline_weighted_score(e, 0.34f, 0.00f, 0.12f, 0.24f, 0.08f, 0.22f);
    }
    return offline_weighted_score(e, 0.30f, 0.15f, 0.15f, 0.15f, 0.10f, 0.15f);
}

static const char *offline_tip_for_eval(const std::string &action, const offline_frame_eval_t &e)
{
    if (e.depth < 58) {
        if (action == "pullup") {
            return "拉起高度不足，下一次胸口主动靠近横杆。";
        }
        if (action == "pushup") {
            return "动作幅度不足，下一次胸口再低一点。";
        }
        if (action == "deadlift") {
            return "髋部折叠不足，下一次先髋后膝。";
        }
        return "动作幅度不足，下一次把行程做完整。";
    }
    if (e.knee < 58) {
        if (action == "squat") {
            return "膝盖轨迹不稳，注意膝盖和脚尖同向。";
        }
        if (action == "pullup" || action == "pushup" || action == "bench_press" || action == "curl" || action == "dumbbell_curl") {
            return "左右发力不同步，下一次放慢并稳定两侧。";
        }
        return "关节节奏不稳，下一次放慢速度。";
    }
    if (e.torso < 58) {
        if (action == "deadlift") {
            return "背部稳定不足，下一次抬胸并收紧核心。";
        }
        if (action == "plank") {
            return "核心支撑不稳，保持肩髋踝一条线。";
        }
        return "躯干稳定不足，下一次收紧核心。";
    }
    if (e.balance < 56) {
        return "稳定性略低，注意左右同步和身体控制。";
    }
    if (e.track < 55) {
        return "识别质量偏低，尽量让全身完整入镜。";
    }
    return "动作完成，继续保持稳定节奏。";
}

static offline_frame_eval_t offline_average_eval_range(const std::vector<offline_frame_eval_t> &evals,
                                                       int start,
                                                       int end,
                                                       const std::string &action)
{
    offline_frame_eval_t out = {};
    if (evals.empty()) {
        return out;
    }
    start = std::max(0, std::min(start, (int)evals.size() - 1));
    end = std::max(start, std::min(end, (int)evals.size() - 1));
    int n = 0;
    int depth_max = 0;
    int knee_sum = 0;
    int hip_sum = 0;
    int torso_sum = 0;
    int balance_sum = 0;
    int track_sum = 0;
    float signal_max = 0.0f;
    for (int i = start; i <= end; ++i) {
        const offline_frame_eval_t &e = evals[i];
        signal_max = std::max(signal_max, e.signal);
        depth_max = std::max(depth_max, e.depth);
        knee_sum += e.knee;
        hip_sum += e.hip;
        torso_sum += e.torso;
        balance_sum += e.balance;
        track_sum += e.track;
        n++;
    }
    if (n <= 0) {
        return out;
    }
    out.signal = signal_max;
    out.depth = depth_max;
    out.knee = knee_sum / n;
    out.hip = hip_sum / n;
    out.torso = torso_sum / n;
    out.balance = balance_sum / n;
    out.track = track_sum / n;
    out.total = offline_weighted_score_for_action(action, out);
    return out;
}

static int offline_count_cycles_with_report(const std::vector<offline_frame_eval_t> &evals,
                                            const video_item_t &item,
                                            std::string *report,
                                            offline_analysis_result_t *rep_avg = nullptr)
{
    if (report) {
        report->clear();
    }
    if (rep_avg) {
        rep_avg->count = 0;
        rep_avg->rep_score = 0;
        rep_avg->depth = 0;
        rep_avg->knee = 0;
        rep_avg->hip = 0;
        rep_avg->torso = 0;
        rep_avg->balance = 0;
        rep_avg->track = 0;
    }
    if (evals.size() < 4) {
        if (report) {
            *report = "未形成完整动作周期：有效动作帧不足。";
        }
        return 0;
    }

    const bool is_pullup = item.action == "pullup";
    const bool is_push_or_bench = item.action == "pushup" || item.action == "bench_press";
    const bool is_upper_cycle = is_pullup || is_push_or_bench;
    const bool is_lower_cycle = item.action == "squat" || item.action == "deadlift";

    std::vector<float> raw_signal(evals.size(), 0.0f);
    for (int i = 0; i < (int)evals.size(); ++i) {
        raw_signal[i] = evals[i].signal;
    }

    auto median3 = [](float a, float b, float c) {
        if (a > b) {
            std::swap(a, b);
        }
        if (b > c) {
            std::swap(b, c);
        }
        if (a > b) {
            std::swap(a, b);
        }
        return b;
    };

    std::vector<float> denoised(evals.size(), 0.0f);
    for (int i = 0; i < (int)evals.size(); ++i) {
        const float p = raw_signal[std::max(0, i - 1)];
        const float c = raw_signal[i];
        const float n = raw_signal[std::min((int)evals.size() - 1, i + 1)];
        const float m = median3(p, c, n);
        denoised[i] = fabsf(c - m) > 0.22f ? m : c;
    }

    std::vector<float> signal(evals.size(), 0.0f);
    for (int i = 0; i < (int)evals.size(); ++i) {
        const float p1 = denoised[std::max(0, i - 1)];
        const float cur = denoised[i];
        const float n1 = denoised[std::min((int)evals.size() - 1, i + 1)];
        signal[i] = p1 * 0.22f + cur * 0.56f + n1 * 0.22f;
    }

    auto percentile = [](std::vector<float> values, float q) {
        if (values.empty()) {
            return 0.0f;
        }
        std::sort(values.begin(), values.end());
        const float pos = std::max(0.0f, std::min(1.0f, q)) * (float)(values.size() - 1);
        const int lo = (int)floorf(pos);
        const int hi = std::min((int)values.size() - 1, lo + 1);
        const float frac = pos - (float)lo;
        return values[lo] * (1.0f - frac) + values[hi] * frac;
    };

    float min_v = percentile(signal, 0.15f);
    float max_v = percentile(signal, 0.85f);
    float range = max_v - min_v;
    if (range < 0.03f) {
        min_v = *std::min_element(signal.begin(), signal.end());
        max_v = *std::max_element(signal.begin(), signal.end());
        range = max_v - min_v;
    }
    float high_frac = 0.60f;
    float low_frac = 0.38f;
    float return_frac = 0.42f;
    float rearm_frac = low_frac;
    float peak_drop_frac = 0.30f;
    float min_rep_range = 0.10f;
    int min_cycle_samples = 4;
    int min_gap_samples = 4;
    int min_peak_samples = 1;
    int min_return_samples = 2;
    const bool peak_drop_count = is_upper_cycle;
    if (is_lower_cycle) {
        high_frac = 0.56f;
        low_frac = 0.30f;
        return_frac = 0.36f;
        rearm_frac = 0.36f;
        peak_drop_frac = 0.32f;
        min_rep_range = 0.075f;
        min_cycle_samples = 4;
        min_gap_samples = 4;
        min_peak_samples = 1;
        min_return_samples = 2;
    } else if (is_pullup) {
        high_frac = 0.50f;
        low_frac = 0.30f;
        return_frac = 0.38f;
        rearm_frac = 0.34f;
        peak_drop_frac = 0.34f;
        min_rep_range = 0.055f;
        min_cycle_samples = 3;
        min_gap_samples = 4;
        min_peak_samples = 1;
        min_return_samples = 1;
    } else if (is_push_or_bench) {
        high_frac = 0.50f;
        low_frac = 0.30f;
        return_frac = 0.38f;
        rearm_frac = 0.34f;
        peak_drop_frac = 0.34f;
        min_rep_range = 0.055f;
        min_cycle_samples = 3;
        min_gap_samples = 4;
        min_peak_samples = 1;
        min_return_samples = 1;
    } else if (item.action == "situp" || item.action == "dumbbell_curl" || item.action == "curl") {
        high_frac = 0.56f;
        low_frac = 0.30f;
        return_frac = 0.36f;
        rearm_frac = 0.36f;
        peak_drop_frac = 0.32f;
        min_rep_range = 0.075f;
        min_cycle_samples = 4;
        min_gap_samples = 4;
        min_peak_samples = 1;
        min_return_samples = 2;
    }
    if (range < min_rep_range) {
        if (report) {
            *report = "未形成完整动作周期：动作幅度变化偏小。";
        }
        return 0;
    }

    const float high = min_v + range * high_frac;
    const float low = min_v + range * low_frac;
    const float ret = min_v + range * return_frac;
    (void)rearm_frac;
    (void)peak_drop_frac;
    (void)min_peak_samples;
    (void)min_return_samples;
    (void)peak_drop_count;
    int count = 0;
    int score_sum = 0;
    int depth_sum = 0;
    int knee_sum = 0;
    int hip_sum = 0;
    int torso_sum = 0;
    int balance_sum = 0;
    int track_sum = 0;
    std::string lines;
    char line[256];

    std::vector<std::pair<int, int>> rep_windows;
    auto add_rep = [&](int start, int end) -> bool {
        start = std::max(0, std::min(start, (int)evals.size() - 1));
        end = std::max(start, std::min(end, (int)evals.size() - 1));
        if (end - start + 1 < min_cycle_samples) {
            return false;
        }
        const offline_frame_eval_t rep = offline_average_eval_range(evals, start, end, item.action);
        count++;
        rep_windows.push_back({start, end});
        score_sum += rep.total;
        depth_sum += rep.depth;
        knee_sum += rep.knee;
        hip_sum += rep.hip;
        torso_sum += rep.torso;
        balance_sum += rep.balance;
        track_sum += rep.track;
        snprintf(line,
                 sizeof(line),
                 "rep %d: score %d, depth %d knee %d hip %d torso %d stable %d track %d, %s\n",
                 count,
                 rep.total,
                 rep.depth,
                 rep.knee,
                 rep.hip,
                 rep.torso,
                 rep.balance,
                 rep.track,
                  offline_tip_for_eval(item.action, rep));
        lines += line;
        return true;
    };

    enum {
        PHASE_UNKNOWN = -1,
        PHASE_LOW = 0,
        PHASE_HIGH = 1,
    };
    typedef struct {
        int index;
        int state;
    } offline_phase_transition_t;

    const int stable_needed = 2;
    const int min_phase_samples = is_lower_cycle ? 3 : 2;
    std::vector<offline_phase_transition_t> transitions;
    int current_phase = PHASE_UNKNOWN;
    int initial_phase = PHASE_UNKNOWN;
    int phase_start = 0;
    int candidate_phase = PHASE_UNKNOWN;
    int candidate_first = 0;
    int candidate_count = 0;
    int last_transition_index = -1000;

    auto phase_for_value = [&](float v) {
        if (v >= high) {
            return PHASE_HIGH;
        }
        if (v <= low) {
            return PHASE_LOW;
        }
        return PHASE_UNKNOWN;
    };

    for (int i = 0; i < (int)signal.size(); ++i) {
        const float v = signal[i];
        const int phase = phase_for_value(v);
        if (phase == PHASE_UNKNOWN) {
            candidate_phase = PHASE_UNKNOWN;
            candidate_count = 0;
            continue;
        }
        if (current_phase == PHASE_UNKNOWN) {
            current_phase = phase;
            initial_phase = phase;
            phase_start = i;
            candidate_phase = PHASE_UNKNOWN;
            candidate_count = 0;
            continue;
        }
        if (phase == current_phase) {
            candidate_phase = PHASE_UNKNOWN;
            candidate_count = 0;
            continue;
        }
        if (candidate_phase != phase) {
            candidate_phase = phase;
            candidate_first = i;
            candidate_count = 1;
        } else {
            candidate_count++;
        }
        const bool stable_phase = candidate_count >= stable_needed;
        const bool long_enough = candidate_first - phase_start >= min_phase_samples;
        const bool gap_ok = candidate_first - last_transition_index >= min_gap_samples;
        if (stable_phase && long_enough && gap_ok) {
            transitions.push_back({candidate_first, phase});
            current_phase = phase;
            phase_start = candidate_first;
            last_transition_index = candidate_first;
            candidate_phase = PHASE_UNKNOWN;
            candidate_count = 0;
        }
    }

#if 0
    for (int i = 0; i < (int)signal.size(); ++i) {
        const float v = signal[i];
        if (!in_peak) {
            if (v <= rearm || i == 0) {
                armed = true;
                cycle_start = i;
                cycle_low = v;
            } else {
                cycle_low = std::min(cycle_low, v);
            }
            if (armed && v >= high && i - last_count_index >= min_gap_samples) {
                in_peak = true;
                peak_start = std::max(0, cycle_start);
                peak_value = v;
                peak_samples = 1;
                return_samples = 0;
            }
            continue;
        }

        if (v >= high) {
            peak_samples++;
        }
        if (v >= peak_value) {
            peak_value = v;
        }
        if (v <= ret || (peak_drop_count && peak_value - v >= range * peak_drop_frac)) {
            return_samples++;
        } else {
            return_samples = 0;
        }

        const bool enough_peak = peak_samples >= min_peak_samples;
        const bool enough_return = return_samples >= min_return_samples;
        const bool enough_cycle = i - peak_start + 1 >= min_cycle_samples;
        const bool enough_gap = i - last_count_index >= min_gap_samples;
        const bool enough_range = peak_value - cycle_low >= std::max(min_rep_range, range * 0.42f);
        if (enough_peak && enough_return && enough_cycle && enough_gap && enough_range) {
                const offline_frame_eval_t rep = offline_average_eval_range(evals, peak_start, i, item.action);
                count++;
                score_sum += rep.total;
                depth_sum += rep.depth;
                knee_sum += rep.knee;
                hip_sum += rep.hip;
                torso_sum += rep.torso;
                balance_sum += rep.balance;
                track_sum += rep.track;
                snprintf(line,
                         sizeof(line),
                         "第%d次：%d分，幅度%d 膝%d 髋%d 躯干%d 稳定%d 识别%d，%s\n",
                         count,
                         rep.total,
                         rep.depth,
                         rep.knee,
                         rep.hip,
                         rep.torso,
                         rep.balance,
                         rep.track,
                         offline_tip_for_eval(item.action, rep));
                lines += line;
                in_peak = false;
                armed = v <= rearm;
                last_count_index = i;
                cycle_start = i;
                cycle_low = v;
                peak_samples = 0;
                return_samples = 0;
                peak_value = v;
        }
    }

    if (in_peak && peak_start >= 0) {
        const int last = (int)signal.size() - 1;
        const bool enough_peak = peak_samples >= min_peak_samples;
        const bool enough_cycle = last - peak_start + 1 >= min_cycle_samples;
        const bool enough_gap = last - last_count_index >= min_gap_samples;
        const bool enough_range = peak_value - cycle_low >= std::max(min_rep_range, range * 0.42f);
        const bool meaningful_return = peak_value - signal[last] >= range * peak_drop_frac ||
                                       signal[last] <= ret;
        if (enough_peak && enough_cycle && enough_gap && enough_range && meaningful_return) {
            const offline_frame_eval_t rep = offline_average_eval_range(evals, peak_start, last, item.action);
            count++;
            score_sum += rep.total;
            depth_sum += rep.depth;
            knee_sum += rep.knee;
            hip_sum += rep.hip;
            torso_sum += rep.torso;
            balance_sum += rep.balance;
            track_sum += rep.track;
            snprintf(line,
                     sizeof(line),
                     "rep %d: %d, depth %d knee %d hip %d torso %d stable %d track %d, %s\n",
                     count,
                     rep.total,
                     rep.depth,
                     rep.knee,
                     rep.hip,
                     rep.torso,
                     rep.balance,
                     rep.track,
                     offline_tip_for_eval(item.action, rep));
            lines += line;
        }
    }

#endif

    if (item.action == "deadlift" && initial_phase == PHASE_HIGH) {
        int rep_start = 0;
        int last_high_start = 0;
        for (const offline_phase_transition_t &tr : transitions) {
            if (tr.state == PHASE_LOW) {
                add_rep(rep_start, tr.index);
            } else if (tr.state == PHASE_HIGH) {
                last_high_start = tr.index;
            }
            rep_start = tr.index;
        }
        if (!transitions.empty() && transitions.back().state == PHASE_HIGH) {
            add_rep(last_high_start, (int)evals.size() - 1);
        }
    } else {
        const int cycle_pairs = (int)transitions.size() / 2;
        for (int r = 0; r < cycle_pairs; ++r) {
            const int start = (r == 0) ? 0 : transitions[r * 2 - 1].index;
            const int end = transitions[r * 2 + 1].index;
            add_rep(start, end);
        }
        const bool tail_can_count = item.action == "pullup";
        if (tail_can_count &&
            initial_phase == PHASE_LOW &&
            (transitions.size() % 2) == 1 &&
            !transitions.empty() &&
            transitions.back().state == PHASE_HIGH) {
            const int start = transitions.size() >= 2 ? transitions[transitions.size() - 2].index : 0;
            add_rep(start, (int)evals.size() - 1);
        }
    }

    auto detect_motion_extrema = [&](bool want_peak) {
        std::vector<int> out;
        const int n = (int)signal.size();
        const int radius = 2;
        const int search = item.action == "pullup" ? 9 : 7;
        const int min_sep = item.action == "pullup" ? std::max(9, n / 6) : 5;
        const float level_thr = want_peak ? high : low;
        const float prom_thr = range * (item.action == "pullup" ? 0.17f : 0.15f);
        for (int i = radius; i < n - radius; ++i) {
            float center = signal[i];
            int best = i;
            for (int j = i - radius; j <= i + radius; ++j) {
                if ((want_peak && signal[j] > signal[best]) ||
                    (!want_peak && signal[j] < signal[best])) {
                    best = j;
                }
            }
            if (best != i) {
                continue;
            }
            if ((want_peak && center < level_thr) || (!want_peak && center > level_thr)) {
                continue;
            }

            const int l0 = std::max(0, i - search);
            const int r1 = std::min(n - 1, i + search);
            float left_opposite = signal[i];
            float right_opposite = signal[i];
            for (int j = l0; j <= i; ++j) {
                left_opposite = want_peak ? std::min(left_opposite, signal[j]) :
                                            std::max(left_opposite, signal[j]);
            }
            for (int j = i; j <= r1; ++j) {
                right_opposite = want_peak ? std::min(right_opposite, signal[j]) :
                                             std::max(right_opposite, signal[j]);
            }
            const float prominence = want_peak ?
                center - std::max(left_opposite, right_opposite) :
                std::min(left_opposite, right_opposite) - center;
            if (prominence < prom_thr) {
                continue;
            }

            if (!out.empty() && i - out.back() < min_sep) {
                const int prev = out.back();
                if ((want_peak && signal[i] > signal[prev]) ||
                    (!want_peak && signal[i] < signal[prev])) {
                    out.back() = i;
                }
            } else {
                out.push_back(i);
            }
        }
        return out;
    };

    const bool extrema_peak_action = item.action == "squat" || item.action == "pullup";
    const bool extrema_valley_action = item.action == "deadlift";
    if (extrema_peak_action || extrema_valley_action) {
        const bool want_peak = extrema_peak_action;
        const std::vector<int> extrema = detect_motion_extrema(want_peak);
        if (item.action == "pullup" && count > 0 && (int)extrema.size() > count && (int)extrema.size() <= count + 2) {
            const int target = count + 1;
            const int window = 6;
            while (count < target) {
                int best = -1;
                float best_value = -1.0f;
                for (int p : extrema) {
                    bool covered = false;
                    for (const auto &w : rep_windows) {
                        if (p >= w.first && p <= w.second) {
                            covered = true;
                            break;
                        }
                    }
                    if (!covered && signal[p] > best_value) {
                        best = p;
                        best_value = signal[p];
                    }
                }
                if (best < 0) {
                    break;
                }
                if (!add_rep(std::max(0, best - window), std::min((int)evals.size() - 1, best + window))) {
                    break;
                }
            }
        }
        ESP_LOGI(TAG,
                 "Offline count extrema hint: action=%s phases_count=%d extrema=%d",
                 item.action.c_str(),
                 count,
                 (int)extrema.size());
    }

    ESP_LOGI(TAG,
             "Offline count signal: action=%s evals=%d min=%.3f max=%.3f range=%.3f high=%.3f low=%.3f ret=%.3f phases=%d init=%d count=%d",
             item.action.c_str(),
             (int)evals.size(),
             (double)min_v,
             (double)max_v,
             (double)range,
             (double)high,
             (double)low,
             (double)ret,
             (int)transitions.size(),
             initial_phase,
             count);

    if (report) {
        if (lines.empty()) {
            *report = "未形成完整动作周期：请让动作从起始位到结束位更清楚。";
        } else {
            *report = lines;
        }
    }
    if (rep_avg && count > 0) {
        rep_avg->count = count;
        rep_avg->rep_score = score_sum / count;
        rep_avg->depth = depth_sum / count;
        rep_avg->knee = knee_sum / count;
        rep_avg->hip = hip_sum / count;
        rep_avg->torso = torso_sum / count;
        rep_avg->balance = balance_sum / count;
        rep_avg->track = track_sum / count;
    }
    return count;
}

static void offline_fill_rect_rgb565(uint16_t *px,
                                     uint32_t stride,
                                     uint32_t w,
                                     uint32_t h,
                                     int x1,
                                     int y1,
                                     int x2,
                                     int y2,
                                     uint16_t color)
{
    x1 = std::max(0, std::min((int)w - 1, x1));
    y1 = std::max(0, std::min((int)h - 1, y1));
    x2 = std::max(0, std::min((int)w - 1, x2));
    y2 = std::max(0, std::min((int)h - 1, y2));
    if (x2 <= x1 || y2 <= y1) {
        return;
    }
    for (int y = y1; y <= y2; ++y) {
        uint16_t *row = px + y * stride;
        for (int x = x1; x <= x2; ++x) {
            row[x] = color;
        }
    }
}

static void offline_draw_pose_result(dl::image::img_t &img, const pose_result_t &pose, int score)
{
    if (!img.data || img.pix_type != dl::image::DL_IMAGE_PIX_TYPE_RGB565LE || img.width == 0 || img.height == 0) {
        return;
    }
    uint16_t *px = reinterpret_cast<uint16_t *>(img.data);
    const uint32_t w = img.width;
    const uint32_t h = img.height;
    std::vector<pose_result_t> one;
    one.push_back(pose);
    pose_overlay_t overlay = make_pose_overlay(one, w, h, 0.08f, 0.08f);
    draw_pose_overlay_layer(px, w, w, h, overlay, 0x07e0, 0x07e0, 0xffe0);

    const int bar_w = std::max(24, std::min(160, (int)w / 4));
    const int fill_w = (bar_w * std::max(0, std::min(100, score))) / 100;
    const uint16_t score_color = score >= 80 ? 0x07e0 : (score >= 60 ? 0xffe0 : 0xf800);
    offline_fill_rect_rgb565(px, w, w, h, 12, 12, 12 + bar_w, 24, 0x0000);
    offline_fill_rect_rgb565(px, w, w, h, 12, 12, 12 + fill_w, 24, score_color);
    draw_rect_rgb565(px, w, w, h, 12, 12, 12 + bar_w, 24, 0xffff);
}

static uint8_t *offline_load_file_psram(const char *path, size_t size)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return nullptr;
    }
    uint8_t *data = (uint8_t *)heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!data) {
        data = (uint8_t *)heap_caps_malloc(size, MALLOC_CAP_8BIT);
    }
    if (!data) {
        fclose(fp);
        return nullptr;
    }
    const size_t n = fread(data, 1, size, fp);
    fclose(fp);
    if (n != size) {
        heap_caps_free(data);
        return nullptr;
    }
    return data;
}

static void offline_collect_jpeg_frames(const uint8_t *data, size_t len, std::vector<offline_jpeg_frame_t> *frames)
{
    if (!data || !frames) {
        return;
    }
    frames->clear();
    size_t start = SIZE_MAX;
    for (size_t i = 0; i + 1 < len; ++i) {
        if (data[i] == 0xff && data[i + 1] == 0xd8) {
            start = i;
            ++i;
            continue;
        }
        if (start != SIZE_MAX && data[i] == 0xff && data[i + 1] == 0xd9) {
            offline_jpeg_frame_t frame = {};
            frame.start = start;
            frame.len = (i + 2) - start;
            frames->push_back(frame);
            start = SIZE_MAX;
            ++i;
        }
    }
}

static bool offline_write_result(const video_item_t &item, const offline_analysis_result_t &result)
{
    mkdir(VIDEO_ROOT, 0775);
    mkdir(OFFLINE_RESULT_DIR, 0775);
    const std::string path = offline_result_path_for(item);
    FILE *fp = fopen(path.c_str(), "w");
    if (!fp) {
        ESP_LOGW(TAG, "Offline result write failed: %s errno=%d", path.c_str(), errno);
        return false;
    }
    const int profile_index = training_action_index_from_id(item.action);
    const training_profile_t *profile = training_profile_by_index(profile_index);
    const offline_eval_hint_t hint = offline_eval_hint_for_action(item.action);
    fprintf(fp, "title=%s\n", item.title.c_str());
    fprintf(fp, "action=%s\n", item.action.c_str());
    fprintf(fp, "profile=%s\n", profile->name);
    fprintf(fp, "sample_fps=%d\n", OFFLINE_ANALYSIS_SAMPLE_FPS);
    fprintf(fp, "sample_target=%d\n", result.sample_count);
    fprintf(fp, "frame_count=%d\n", result.frame_count);
    fprintf(fp, "valid_frames=%d\n", result.valid_frames);
    fprintf(fp, "elapsed_ms=%d\n", result.elapsed_ms);
    fprintf(fp, "status=%s\n", result.status.empty() ? "local_movenet" : result.status.c_str());
    fprintf(fp, "result_video=%s\n", result.result_video.c_str());
    fprintf(fp, "count=%d\n", result.count);
    fprintf(fp, "score=%d\n", result.score > 0 ? result.score : hint.score);
    fprintf(fp, "rep_score=%d\n", result.rep_score);
    fprintf(fp, "score_depth=%d\n", result.depth > 0 ? result.depth : hint.depth);
    fprintf(fp, "score_knee=%d\n", result.knee > 0 ? result.knee : hint.knee);
    fprintf(fp, "score_hip=%d\n", result.hip > 0 ? result.hip : hint.hip);
    fprintf(fp, "score_torso=%d\n", result.torso > 0 ? result.torso : hint.torso);
    fprintf(fp, "score_balance=%d\n", result.balance > 0 ? result.balance : hint.balance);
    fprintf(fp, "score_track=%d\n", result.track > 0 ? result.track : hint.track);
    fprintf(fp, "tip=%s\n", offline_tip_for_action(item.action));
    fprintf(fp, "rule=%s\n", hint.rule);
    fprintf(fp, "correction=%s\n", hint.correction);
    fprintf(fp, "view=%s\n", hint.view);
    if (!result.rep_log.empty()) {
        std::string log = result.rep_log;
        for (char &ch : log) {
            if (ch == '\r') {
                ch = ' ';
            } else if (ch == '\n') {
                ch = '|';
            }
        }
        fprintf(fp, "rep_log=%s\n", log.c_str());
    }
    fprintf(fp, "note=%s\n", "离线检测已使用板端本地 MoveNet 抽帧推理，并把带骨架的结果视频缓存到 results 目录。");
    fclose(fp);

    training_history_item_t history = {};
    history.valid = true;
    strlcpy(history.source, "offline", sizeof(history.source));
    strlcpy(history.action, profile->name, sizeof(history.action));
    format_record_time(history.time, sizeof(history.time));
    strlcpy(history.tip, hint.correction, sizeof(history.tip));
    history.count = result.count;
    history.target = 0;
    history.score = result.score > 0 ? result.score : hint.score;
    history.rep_score = result.rep_score;
    history.score_depth = result.depth > 0 ? result.depth : hint.depth;
    history.score_knee = result.knee > 0 ? result.knee : hint.knee;
    history.score_hip = result.hip > 0 ? result.hip : hint.hip;
    history.score_torso = result.torso > 0 ? result.torso : hint.torso;
    history.score_balance = result.balance > 0 ? result.balance : hint.balance;
    history.score_track = result.track > 0 ? result.track : hint.track;
    history.elapsed_ms = (uint32_t)std::max(0, result.elapsed_ms);
    if (!result.rep_log.empty()) {
        training_history_sanitize_copy(history.reps, sizeof(history.reps), result.rep_log.c_str());
    } else {
        snprintf(history.reps, sizeof(history.reps), "有效帧 %d/%d", result.valid_frames, result.sample_count);
    }
    training_history_append(history);
    return true;
}

typedef struct {
    video_item_t item;
} offline_process_arg_t;

typedef struct {
    int progress;
    bool done;
    char text[224];
} offline_ui_update_t;

static void offline_ui_update_async(void *user_data)
{
    offline_ui_update_t *update = static_cast<offline_ui_update_t *>(user_data);
    if (!update) {
        return;
    }
    if (s_offline_progress) {
        lv_bar_set_value(s_offline_progress, update->progress, LV_ANIM_ON);
    }
    if (s_offline_status_label && update->text[0]) {
        lv_label_set_text(s_offline_status_label, update->text);
    }
    if (!update->done && s_offline_report_label && update->text[0]) {
        lv_label_set_text(s_offline_report_label, update->text);
    }
    if (update->done) {
        offline_update_selected_status();
        offline_render_list();
        if (lv_scr_act() == scr_correction &&
            s_offline_player_host &&
            s_offline_selected_index >= 0 &&
            s_offline_selected_index < (int)s_offline_videos.size()) {
            const video_item_t play_item = offline_playback_item_for(s_offline_videos[s_offline_selected_index]);
            if (access(play_item.path.c_str(), F_OK) == 0) {
                video_player_play_item(s_offline_player_host, play_item, nullptr);
            }
        }
    }
    delete update;
}

static void offline_post_ui_update(int progress, bool done, const char *fmt, ...)
{
    offline_ui_update_t *update = new (std::nothrow) offline_ui_update_t();
    if (!update) {
        return;
    }
    update->progress = std::max(0, std::min(100, progress));
    update->done = done;
    update->text[0] = '\0';
    if (fmt) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(update->text, sizeof(update->text), fmt, ap);
        va_end(ap);
    }
    lv_async_call(offline_ui_update_async, update);
}

static void offline_video_process_task(void *arg)
{
    offline_process_arg_t *ctx = static_cast<offline_process_arg_t *>(arg);
    if (!ctx) {
        s_offline_process_task = nullptr;
        vTaskDelete(NULL);
        return;
    }

    const int64_t start_us = esp_timer_get_time();
    const video_item_t item = ctx->item;
    delete ctx;

    struct stat st = {};
    if (item.path.empty() || stat(item.path.c_str(), &st) != 0 || st.st_size <= 0) {
        offline_post_ui_update(0, false, "未找到离线视频文件，请检查 /FIT_VIDEO/offline。");
        s_offline_process_task = nullptr;
        vTaskDelete(NULL);
        return;
    }

    offline_post_ui_update(4, false, "正在准备本地姿态模型...");
    if (!lock_models(pdMS_TO_TICKS(6000))) {
        offline_post_ui_update(0, false, "本地模型正忙，请稍后重试。");
        s_offline_process_task = nullptr;
        vTaskDelete(NULL);
        return;
    }
    if (!s_pose_det) {
        s_pose_det = new (std::nothrow) HumanPoseDetect(0.16f, 0.70f, 1);
    }
    const bool pose_ready = s_pose_det && s_pose_det->ready();
    const char *pose_error = s_pose_det ? s_pose_det->error() : "内存不足";
    unlock_models();
    if (!pose_ready) {
        offline_post_ui_update(0, false, "本地姿态模型初始化失败：%s", pose_error ? pose_error : "unknown");
        s_offline_process_task = nullptr;
        vTaskDelete(NULL);
        return;
    }

    offline_post_ui_update(8, false, "正在读取视频：%.1fMB", (double)st.st_size / (1024.0 * 1024.0));
    uint8_t *video_data = offline_load_file_psram(item.path.c_str(), (size_t)st.st_size);
    if (!video_data) {
        offline_post_ui_update(0, false, "视频读入内存失败，文件可能过大或内存不足。");
        s_offline_process_task = nullptr;
        vTaskDelete(NULL);
        return;
    }

    std::vector<offline_jpeg_frame_t> frames;
    offline_collect_jpeg_frames(video_data, (size_t)st.st_size, &frames);
    if (frames.empty()) {
        heap_caps_free(video_data);
        offline_post_ui_update(0, false, "视频帧解析失败，请确认是 MJPEG 文件。");
        ESP_LOGW(TAG, "Offline collect failed: %s errno=%d", item.path.c_str(), errno);
        s_offline_process_task = nullptr;
        vTaskDelete(NULL);
        return;
    }

    offline_analysis_result_t result = {};
    result.frame_count = (int)frames.size();
    result.sample_count = std::max(1, std::min(OFFLINE_ANALYSIS_MAX_SAMPLES, result.frame_count));
    result.result_video = offline_result_video_path_for(item);
    result.status = "local_movenet";
    ESP_LOGI(TAG,
             "Offline local MoveNet start: title=%s file=%s bytes=%ld frames=%d samples=%d out=%s",
             item.title.c_str(),
             item.path.c_str(),
             (long)st.st_size,
             result.frame_count,
             result.sample_count,
             result.result_video.c_str());

    mkdir(VIDEO_ROOT, 0775);
    mkdir(OFFLINE_RESULT_DIR, 0775);
    FILE *out_fp = fopen(result.result_video.c_str(), "wb");
    if (!out_fp) {
        heap_caps_free(video_data);
        offline_post_ui_update(0, false, "结果视频创建失败，请检查 SD 卡可写状态。");
        ESP_LOGW(TAG, "Offline result video open failed: %s errno=%d", result.result_video.c_str(), errno);
        s_offline_process_task = nullptr;
        vTaskDelete(NULL);
        return;
    }

    std::vector<offline_frame_eval_t> frame_evals;
    frame_evals.reserve(result.sample_count);
    int encoded_frames = 0;
    int last_score = 0;
    int pose_frames = 0;
    int eval_fail_frames = 0;
    int no_pose_frames = 0;
    reset_offline_analysis_smooth();
    reset_offline_draw_smooth();
    for (int i = 0; i < result.sample_count; ++i) {
        const int frame_index = result.sample_count > 1 ?
            (int)(((int64_t)i * (int64_t)(result.frame_count - 1)) / (int64_t)(result.sample_count - 1)) : 0;
        const offline_jpeg_frame_t &frame = frames[frame_index];
        dl::image::jpeg_img_t jpeg_img = {
            .data = (void *)(video_data + frame.start),
            .data_len = frame.len,
        };

        dl::image::img_t img = dl::image::hw_decode_jpeg(jpeg_img, dl::image::DL_IMAGE_PIX_TYPE_RGB565LE, 500);
        if (!img.data || img.width == 0 || img.height == 0) {
            ESP_LOGW(TAG, "Offline JPEG decode failed frame=%d len=%u", frame_index, (unsigned)frame.len);
            continue;
        }

        std::vector<pose_result_t> detected;
        int valid_count = 0;
        float avg_score = 0.0f;
        if (lock_models(pdMS_TO_TICKS(5000))) {
            if (s_pose_det && s_pose_det->ready()) {
                const std::vector<pose_result_t> &model_results = s_pose_det->run(img);
                detected = model_results;
            }
            unlock_models();
        }

        pose_result_t best = {};
        if (offline_best_pose(detected, &best, &valid_count, &avg_score)) {
            pose_frames++;
            const pose_result_t analysis_pose = smooth_offline_pose_for_analysis(best, item.action);
            valid_count = offline_pose_valid_count(analysis_pose, &avg_score);
            offline_frame_eval_t eval = {};
            if (offline_eval_action_frame(item, analysis_pose, valid_count, avg_score, &eval)) {
                frame_evals.push_back(eval);
                result.valid_frames++;
                result.depth += eval.depth;
                result.knee += eval.knee;
                result.hip += eval.hip;
                result.torso += eval.torso;
                result.balance += eval.balance;
                result.track += eval.track;
                last_score = eval.total;
            } else {
                eval_fail_frames++;
            }
            const pose_result_t draw_pose = smooth_offline_pose_for_draw(analysis_pose);
            offline_draw_pose_result(img, draw_pose, last_score);
        } else {
            no_pose_frames++;
        }

        dl::image::jpeg_img_t encoded = dl::image::sw_encode_jpeg(img, 74);
        if (encoded.data && encoded.data_len > 0) {
            fwrite(encoded.data, 1, encoded.data_len, out_fp);
            encoded_frames++;
            heap_caps_free(encoded.data);
        } else {
            ESP_LOGW(TAG, "Offline JPEG encode failed sample=%d frame=%d", i, frame_index);
        }
        heap_caps_free(img.data);

        if ((i % 2) == 0 || i == result.sample_count - 1) {
            const int progress = 12 + ((i + 1) * 82) / result.sample_count;
            offline_post_ui_update(progress,
                                   false,
                                   "本地模型检测 %s：%d/%d 帧，有效 %d 帧",
                                   item.title.c_str(),
                                   i + 1,
                                   result.sample_count,
                                   result.valid_frames);
        }
        vTaskDelay(pdMS_TO_TICKS(1));
    }
    fclose(out_fp);
    heap_caps_free(video_data);

    ESP_LOGI(TAG,
             "Offline sample stats: action=%s samples=%d pose=%d eval=%d eval_fail=%d no_pose=%d",
             item.action.c_str(),
             result.sample_count,
             pose_frames,
             result.valid_frames,
             eval_fail_frames,
             no_pose_frames);

    result.elapsed_ms = (int)((esp_timer_get_time() - start_us) / 1000);
    if (encoded_frames <= 0) {
        unlink(result.result_video.c_str());
        result.result_video.clear();
        result.status = "local_movenet_no_video";
    }

    const int denom = std::max(1, result.valid_frames);
    if (item.action == "plank") {
        result.count = result.valid_frames * 1000 / std::max(1, OFFLINE_ANALYSIS_SAMPLE_FPS);
        result.depth = result.depth > 0 ? result.depth / denom : 0;
        result.knee = result.knee > 0 ? result.knee / denom : 0;
        result.hip = result.hip > 0 ? result.hip / denom : 0;
        result.torso = result.torso > 0 ? result.torso / denom : 0;
        result.balance = result.balance > 0 ? result.balance / denom : 0;
        result.track = result.track > 0 ? result.track / denom : 0;
        const int plank_valid_ratio = offline_score_from_unit((float)result.valid_frames / (float)std::max(1, result.sample_count));
        result.track = (result.track * 70 + plank_valid_ratio * 30) / 100;
        offline_frame_eval_t plank_eval = {};
        plank_eval.depth = result.depth;
        plank_eval.knee = result.knee;
        plank_eval.hip = result.hip;
        plank_eval.torso = result.torso;
        plank_eval.balance = result.balance;
        plank_eval.track = result.track;
        result.rep_score = offline_weighted_score_for_action(item.action, plank_eval);
        result.score = result.rep_score;
        char plank_log[160];
        snprintf(plank_log,
                 sizeof(plank_log),
                 "平板支撑：有效支撑约%d秒，综合评分%d分。",
                 result.count / 1000,
                 result.rep_score);
        result.rep_log = plank_log;
    } else {
        offline_analysis_result_t rep_avg = {};
        result.count = offline_count_cycles_with_report(frame_evals, item, &result.rep_log, &rep_avg);
        if (result.count > 0) {
            result.depth = rep_avg.depth;
            result.knee = rep_avg.knee;
            result.hip = rep_avg.hip;
            result.torso = rep_avg.torso;
            result.balance = rep_avg.balance;
            result.track = rep_avg.track;
            result.rep_score = rep_avg.rep_score;
        } else {
            result.depth = result.depth > 0 ? result.depth / denom : 0;
            result.knee = result.knee > 0 ? result.knee / denom : 0;
            result.hip = result.hip > 0 ? result.hip / denom : 0;
            result.torso = result.torso > 0 ? result.torso / denom : 0;
            result.balance = result.balance > 0 ? result.balance / denom : 0;
            result.track = result.track > 0 ? result.track / denom : 0;
            result.rep_score = 0;
        }
    }
    if (item.action != "plank") {
        const int valid_ratio = offline_score_from_unit((float)result.valid_frames / (float)std::max(1, result.sample_count));
        if (result.count > 0) {
            result.track = std::max(result.track, (result.track * 85 + valid_ratio * 15) / 100);
            result.score = result.rep_score;
        } else {
            result.track = (result.track * 70 + valid_ratio * 30) / 100;
            offline_frame_eval_t final_eval = {};
            final_eval.depth = result.depth;
            final_eval.knee = result.knee;
            final_eval.hip = result.hip;
            final_eval.torso = result.torso;
            final_eval.balance = result.balance;
            final_eval.track = result.track;
            result.score = offline_weighted_score_for_action(item.action, final_eval);
        }
    }
    if (result.valid_frames == 0) {
        result.score = 0;
        result.rep_score = 0;
    }

    offline_write_result(item, result);
    offline_post_ui_update(100,
                           true,
                           "检测完成：本地模型 %d/%d 帧，有效 %d 帧，用时 %.1fs。",
                           result.sample_count,
                           result.frame_count,
                           result.valid_frames,
                           (double)result.elapsed_ms / 1000.0);
    ESP_LOGI(TAG,
             "Offline local MoveNet result: title=%s action=%s frames=%d samples=%d valid=%d encoded=%d count=%d score=%d elapsed=%dms result=%s",
             item.title.c_str(),
             item.action.c_str(),
             result.frame_count,
             result.sample_count,
             result.valid_frames,
             encoded_frames,
             result.count,
             result.score,
             result.elapsed_ms,
             result.result_video.c_str());

    s_offline_process_task = nullptr;
    vTaskDelete(NULL);
}

static void course_render_list(void)
{
    if (!s_course_list) {
        return;
    }
    lv_obj_clean(s_course_list);
    if (!s_sdcard_mounted) {
        make_text(s_course_list, "等待 SD 卡挂载，稍后点重新扫描。", 0, 0, 280, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
        return;
    }
    video_load_manifest(COURSE_VIDEO_MANIFEST, COURSE_VIDEO_DIR, &s_course_videos);
    if (s_course_videos.empty()) {
        make_text(s_course_list, "没有读取到课程视频，请检查 /FIT_VIDEO/courses/manifest.csv。", 0, 0, 300, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
        return;
    }
    for (int i = 0; i < (int)s_course_videos.size(); ++i) {
        const int y = i * 44;
        make_product_button(s_course_list,
                            0,
                            y,
                            276,
                            36,
                            s_course_videos[i].title.c_str(),
                            course_video_select_cb,
                            i == s_course_selected_index,
                            (void *)(intptr_t)i);
    }
}

static void offline_render_list(void)
{
    if (!s_offline_list) {
        return;
    }
    lv_obj_clean(s_offline_list);
    if (!s_sdcard_mounted) {
        make_text(s_offline_list, "等待 SD 卡挂载，稍后点重新扫描。", 0, 0, 280, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
        return;
    }
    video_load_manifest(OFFLINE_VIDEO_MANIFEST, OFFLINE_VIDEO_DIR, &s_offline_videos);
    if (s_offline_videos.empty()) {
        make_text(s_offline_list, "没有读取到离线视频，请检查 /FIT_VIDEO/offline/manifest.csv。", 0, 0, 300, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
        return;
    }
    for (int i = 0; i < (int)s_offline_videos.size(); ++i) {
        const video_item_t &item = s_offline_videos[i];
        const bool done = access(offline_result_path_for(item).c_str(), F_OK) == 0;
        char label[96];
        snprintf(label, sizeof(label), "%s %s", done ? "已检测" : "待检测", item.title.c_str());
        make_product_button(s_offline_list,
                            0,
                            i * 44,
                            292,
                            36,
                            label,
                            offline_video_select_cb,
                            i == s_offline_selected_index,
                            (void *)(intptr_t)i);
    }
}

static void course_video_refresh_cb(lv_event_t *e)
{
    (void)e;
    if (!s_sdcard_mounted) {
        start_sdcard_retry_task();
    }
    course_render_list();
    if (s_course_status_label) {
        lv_label_set_text(s_course_status_label, s_course_videos.empty() ? "未读取到课程视频。" : "课程列表已刷新。");
    }
}

static void apply_course_video_zoom(bool zoom)
{
    s_course_video_zoomed = zoom;
    if (!s_course_player_card || !s_course_player_host) {
        return;
    }
    if (zoom) {
        lv_obj_set_pos(s_course_player_card, 24, 74);
        lv_obj_set_size(s_course_player_card, 768, 472);
        lv_obj_move_foreground(s_course_player_card);
        if (s_course_title_label) {
            lv_obj_set_pos(s_course_title_label, 0, 36);
            lv_obj_set_width(s_course_title_label, 728);
        }
        lv_obj_set_pos(s_course_player_host, 0, 56);
        lv_obj_set_size(s_course_player_host, 728, 370);
        if (s_course_status_label) {
            lv_obj_set_pos(s_course_status_label, 0, 434);
            lv_obj_set_width(s_course_status_label, 728);
        }
        set_button_text(s_course_zoom_button, "缩小");
    } else {
        lv_obj_set_pos(s_course_player_card, 368, 98);
        lv_obj_set_size(s_course_player_card, 424, 330);
        if (s_course_title_label) {
            lv_obj_set_pos(s_course_title_label, 0, 36);
            lv_obj_set_width(s_course_title_label, 384);
        }
        lv_obj_set_pos(s_course_player_host, 0, 56);
        lv_obj_set_size(s_course_player_host, 384, 216);
        if (s_course_status_label) {
            lv_obj_set_pos(s_course_status_label, 0, 286);
            lv_obj_set_width(s_course_status_label, 380);
        }
        set_button_text(s_course_zoom_button, "放大");
    }
    lv_obj_update_layout(s_course_player_card);
    lv_obj_update_layout(s_course_player_host);
    video_player_resize_current(s_course_player_host);
}

static void course_video_zoom_cb(lv_event_t *e)
{
    (void)e;
    apply_course_video_zoom(!s_course_video_zoomed);
}

static void course_video_select_cb(lv_event_t *e)
{
    const int index = (int)(intptr_t)lv_event_get_user_data(e);
    if (index < 0 || index >= (int)s_course_videos.size()) {
        return;
    }
    s_course_selected_index = index;
    const video_item_t &item = s_course_videos[index];
    if (s_course_title_label) {
        lv_label_set_text(s_course_title_label, item.title.c_str());
    }
    video_player_play_item(s_course_player_host, item, s_course_status_label);
    course_render_list();
}

static void apply_offline_video_zoom(bool zoom)
{
    s_offline_video_zoomed = zoom;
    if (!s_offline_player_card || !s_offline_player_host) {
        return;
    }
    if (zoom) {
        lv_obj_set_pos(s_offline_player_card, 24, 74);
        lv_obj_set_size(s_offline_player_card, 768, 472);
        lv_obj_move_foreground(s_offline_player_card);
        if (s_offline_title_label) {
            lv_obj_set_pos(s_offline_title_label, 0, 36);
            lv_obj_set_width(s_offline_title_label, 728);
        }
        lv_obj_set_pos(s_offline_player_host, 0, 56);
        lv_obj_set_size(s_offline_player_host, 728, 370);
        set_button_text(s_offline_zoom_button, "缩小");
    } else {
        lv_obj_set_pos(s_offline_player_card, 368, 98);
        lv_obj_set_size(s_offline_player_card, 424, 330);
        if (s_offline_title_label) {
            lv_obj_set_pos(s_offline_title_label, 0, 36);
            lv_obj_set_width(s_offline_title_label, 384);
        }
        lv_obj_set_pos(s_offline_player_host, 0, 56);
        lv_obj_set_size(s_offline_player_host, 384, 216);
        set_button_text(s_offline_zoom_button, "放大");
    }
    lv_obj_update_layout(s_offline_player_card);
    lv_obj_update_layout(s_offline_player_host);
    video_player_resize_current(s_offline_player_host);
}

static void offline_video_zoom_cb(lv_event_t *e)
{
    (void)e;
    apply_offline_video_zoom(!s_offline_video_zoomed);
}

static void offline_update_selected_status(void)
{
    if (s_offline_selected_index < 0 || s_offline_selected_index >= (int)s_offline_videos.size()) {
        if (s_offline_title_label) {
            lv_label_set_text(s_offline_title_label, "请选择一个离线视频");
        }
        if (s_offline_status_label) {
            lv_label_set_text(s_offline_status_label, "结果会缓存到 /FIT_VIDEO/results。");
        }
        if (s_offline_report_label) {
            lv_label_set_text(s_offline_report_label, "请选择一个离线视频。");
        }
        if (s_offline_progress) {
            lv_bar_set_value(s_offline_progress, 0, LV_ANIM_OFF);
        }
        return;
    }
    const video_item_t &item = s_offline_videos[s_offline_selected_index];
    if (s_offline_title_label) {
        lv_label_set_text(s_offline_title_label, item.title.c_str());
    }
    std::string result;
    if (offline_load_result(item, &result)) {
        const std::string display = offline_format_result_summary(result);
        if (s_offline_status_label) {
            lv_label_set_text(s_offline_status_label, "已检测，结果已缓存。");
        }
        if (s_offline_report_label) {
            lv_label_set_text(s_offline_report_label, display.c_str());
        }
        if (s_offline_progress) {
            lv_bar_set_value(s_offline_progress, 100, LV_ANIM_OFF);
        }
    } else {
        if (s_offline_status_label) {
            lv_label_set_text(s_offline_status_label, "未检测。点击开始检测后生成摘要结果。");
        }
        if (s_offline_report_label) {
            lv_label_set_text(s_offline_report_label, "暂无报告。检测完成后会显示次数、评分和每次动作纠错提示。");
        }
        if (s_offline_progress) {
            lv_bar_set_value(s_offline_progress, 0, LV_ANIM_OFF);
        }
    }
}

static void offline_video_select_cb(lv_event_t *e)
{
    const int index = (int)(intptr_t)lv_event_get_user_data(e);
    if (index < 0 || index >= (int)s_offline_videos.size()) {
        return;
    }
    s_offline_selected_index = index;
    const video_item_t &item = s_offline_videos[index];
    const video_item_t play_item = offline_playback_item_for(item);
    video_player_play_item(s_offline_player_host, play_item, nullptr);
    offline_update_selected_status();
    offline_render_list();
}

static void offline_video_process_cb(lv_event_t *e)
{
    (void)e;
    if (s_offline_process_task) {
        if (s_offline_status_label) {
            lv_label_set_text(s_offline_status_label, "离线处理正在进行，请稍候。");
        }
        return;
    }
    if (s_offline_selected_index < 0 || s_offline_selected_index >= (int)s_offline_videos.size()) {
        if (s_offline_status_label) {
            lv_label_set_text(s_offline_status_label, "请先选择一个离线视频。");
        }
        return;
    }
    const video_item_t &item = s_offline_videos[s_offline_selected_index];
    struct stat st = {};
    if (item.path.empty() || stat(item.path.c_str(), &st) != 0 || st.st_size <= 0) {
        if (s_offline_status_label) {
            lv_label_set_text(s_offline_status_label, "未找到离线视频文件，请检查 /FIT_VIDEO/offline。");
        }
        if (s_offline_progress) {
            lv_bar_set_value(s_offline_progress, 0, LV_ANIM_OFF);
        }
        ESP_LOGW(TAG, "Offline video missing: %s errno=%d", item.path.c_str(), errno);
        return;
    }
    if (!video_player_stop_current()) {
        if (s_offline_status_label) {
            lv_label_set_text(s_offline_status_label, "视频仍在停止中，请稍后再开始检测。");
        }
        return;
    }
    if (s_offline_progress) {
        lv_bar_set_value(s_offline_progress, 4, LV_ANIM_ON);
    }
    if (s_offline_status_label) {
        char text[180];
        snprintf(text,
                 sizeof(text),
                 "正在读取视频：%.1fMB，目标采样 %dFPS，请稍等。",
                 (double)st.st_size / (1024.0 * 1024.0),
                 OFFLINE_ANALYSIS_SAMPLE_FPS);
        lv_label_set_text(s_offline_status_label, text);
    }
    offline_process_arg_t *arg = new (std::nothrow) offline_process_arg_t();
    if (!arg) {
        if (s_offline_status_label) {
            lv_label_set_text(s_offline_status_label, "离线处理任务创建失败，内存不足。");
        }
        return;
    }
    arg->item = item;
    if (xTaskCreate(offline_video_process_task, "offline_video", 10 * 1024, arg, 4, &s_offline_process_task) != pdPASS) {
        delete arg;
        s_offline_process_task = nullptr;
        if (s_offline_status_label) {
            lv_label_set_text(s_offline_status_label, "离线处理任务创建失败。");
        }
    }
}

static void offline_video_delete_result_cb(lv_event_t *e)
{
    (void)e;
    if (s_offline_selected_index < 0 || s_offline_selected_index >= (int)s_offline_videos.size()) {
        return;
    }
    const video_item_t &item = s_offline_videos[s_offline_selected_index];
    const std::string path = offline_result_path_for(item);
    const std::string video_path = offline_result_video_path_for(item);
    if (strcmp(s_video_player_path, video_path.c_str()) == 0) {
        video_player_stop_current();
    }
    unlink(path.c_str());
    unlink(video_path.c_str());
    offline_update_selected_status();
    offline_render_list();
}

static void offline_video_refresh_cb(lv_event_t *e)
{
    (void)e;
    if (!s_sdcard_mounted) {
        start_sdcard_retry_task();
    }
    offline_render_list();
    offline_update_selected_status();
}

static void create_correction_page(void)
{
    scr_correction = lv_obj_create(NULL);
    if (!scr_correction) {
        mark_lvgl_build_failed("correction screen");
        return;
    }
    lv_obj_t *main = make_product_shell(scr_correction,
                                        "correction",
                                        "姿态纠正",
                                        "角度、稳定性和错误提示从实时训练中拆分出来。");
    if (!main) {
        return;
    }
    add_top_status(main, "SD视频", "本地检测", "结果缓存");

    lv_obj_t *list_card = make_card(main, 24, 98, 320, 448);
    make_text(list_card, "离线视频", 0, 0, 160, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_product_button(list_card, 196, 0, 92, 32, "重新扫描", offline_video_refresh_cb, false);
    s_offline_list = lv_obj_create(list_card);
    if (s_offline_list) {
        lv_obj_remove_style_all(s_offline_list);
        lv_obj_set_pos(s_offline_list, 0, 48);
        lv_obj_set_size(s_offline_list, 292, 236);
        lv_obj_set_scroll_dir(s_offline_list, LV_DIR_VER);
        lv_obj_set_style_pad_all(s_offline_list, 0, LV_PART_MAIN);
    }
    make_text(list_card, "检测报告", 0, 304, 120, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lv_obj_t *report_box = lv_obj_create(list_card);
    if (report_box) {
        lv_obj_set_pos(report_box, 0, 334);
        lv_obj_set_size(report_box, 292, 96);
        lv_obj_set_scroll_dir(report_box, LV_DIR_VER);
        lv_obj_set_style_bg_color(report_box, lv_color_hex(0xf7fafc), LV_PART_MAIN);
        lv_obj_set_style_border_color(report_box, lv_color_hex(0xd9e2e8), LV_PART_MAIN);
        lv_obj_set_style_border_width(report_box, 1, LV_PART_MAIN);
        lv_obj_set_style_radius(report_box, 8, LV_PART_MAIN);
        lv_obj_set_style_pad_all(report_box, 8, LV_PART_MAIN);
        s_offline_report_label = make_text(report_box,
                                           "暂无报告。",
                                           0,
                                           0,
                                           258,
                                           font_14(),
                                           0x314551,
                                           LV_LABEL_LONG_WRAP);
    }

    lv_obj_t *player_card = make_card(main, 368, 98, 424, 330);
    s_offline_player_card = player_card;
    make_text(player_card, "检测预览", 0, 0, 110, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_product_button(player_card, 152, 0, 68, 32, "暂停", video_player_pause_toggle_cb, false);
    make_product_button(player_card, 228, 0, 58, 32, "停止", video_player_stop_button_cb, false, s_offline_player_host);
    s_offline_zoom_button = make_product_button(player_card, 314, 0, 70, 32, "放大", offline_video_zoom_cb, false);
    s_offline_title_label = make_text(player_card, "请选择一个离线视频", 0, 36, 384, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    s_offline_player_host = lv_obj_create(player_card);
    if (s_offline_player_host) {
        lv_obj_set_pos(s_offline_player_host, 0, 56);
        lv_obj_set_size(s_offline_player_host, 384, 216);
        lv_obj_clear_flag(s_offline_player_host, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_set_style_bg_color(s_offline_player_host, lv_color_black(), LV_PART_MAIN);
        lv_obj_set_style_bg_opa(s_offline_player_host, LV_OPA_COVER, LV_PART_MAIN);
        lv_obj_set_style_border_width(s_offline_player_host, 0, LV_PART_MAIN);
        lv_obj_set_style_pad_all(s_offline_player_host, 0, LV_PART_MAIN);
        lv_obj_set_style_radius(s_offline_player_host, 8, LV_PART_MAIN);
        lv_obj_set_style_clip_corner(s_offline_player_host, true, LV_PART_MAIN);
        make_text(s_offline_player_host, "选择视频后播放预览", 110, 92, 180, font_14(), 0xd7dee8, LV_LABEL_LONG_DOT);
    }

    lv_obj_t *result_card = make_card(main, 368, 450, 424, 96);
    make_text(result_card, "离线检测结果", 0, 0, 180, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_product_button(result_card, 226, 0, 74, 32, "开始检测", offline_video_process_cb, true);
    make_product_button(result_card, 310, 0, 74, 32, "删除结果", offline_video_delete_result_cb, false);
    s_offline_progress = lv_bar_create(result_card);
    if (s_offline_progress) {
        lv_obj_set_pos(s_offline_progress, 0, 44);
        lv_obj_set_size(s_offline_progress, 384, 8);
        lv_bar_set_range(s_offline_progress, 0, 100);
        lv_bar_set_value(s_offline_progress, 0, LV_ANIM_OFF);
    }
    s_offline_status_label = make_text(result_card,
                                       "结果会缓存到 /FIT_VIDEO/results。",
                                       0,
                                       64,
                                       384,
                                       font_14(),
                                       0x314551,
                                       LV_LABEL_LONG_WRAP);
    offline_render_list();
    offline_update_selected_status();
    apply_offline_video_zoom(false);
    return;

    lv_obj_t *analysis_new = make_card(main, 24, 98, 376, 176);
    make_text(analysis_new, "当前评估", 0, 0, 200, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_corr_summary = make_text(analysis_new, "-- | 总分 0 | PC 0.0 FPS", 0, 42, 330, font_16(), 0x145883, LV_LABEL_LONG_DOT);
    lbl_corr_tip = make_text(analysis_new,
                             "开始深蹲训练后显示纠错提示。",
                             0,
                             82,
                             330,
                             font_14(),
                             0x314551,
                             LV_LABEL_LONG_WRAP);

    lv_obj_t *logic_new = make_card(main, 424, 98, 368, 176);
    make_text(logic_new, "判定逻辑", 0, 0, 190, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(logic_new, "计数: 连续下蹲确认 -> 连续站起确认 -> +1", 0, 42, 326, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_text(logic_new, "主指标: 幅度、轨迹、核心、躯干和平衡。", 0, 72, 326, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_text(logic_new, "六项动作分别使用自己的计数和纠错规则。", 0, 102, 326, font_14(), 0x687783, LV_LABEL_LONG_DOT);

    lv_obj_t *depth_new = make_card(main, 24, 300, 376, 70);
    lv_obj_set_style_shadow_width(depth_new, 0, LV_PART_MAIN);
    make_text(depth_new, "幅度", 0, 0, 76, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_corr_depth = make_text(depth_new, "幅度 0 | 比例 0.00 | 动作幅度越完整越好", 84, 4, 260, font_14(), 0x314551, LV_LABEL_LONG_DOT);

    lv_obj_t *knee_new = make_card(main, 424, 300, 368, 70);
    lv_obj_set_style_shadow_width(knee_new, 0, LV_PART_MAIN);
    make_text(knee_new, "膝盖", 0, 0, 76, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_corr_knee = make_text(knee_new, "膝盖轨迹 0 | 膝角 0°", 84, 4, 250, font_14(), 0x314551, LV_LABEL_LONG_DOT);

    lv_obj_t *hip_new = make_card(main, 24, 388, 376, 70);
    lv_obj_set_style_shadow_width(hip_new, 0, LV_PART_MAIN);
    make_text(hip_new, "核心", 0, 0, 76, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_corr_hip = make_text(hip_new, "髋部/核心 0 | 0° | 保持稳定发力", 84, 4, 260, font_14(), 0x314551, LV_LABEL_LONG_DOT);

    lv_obj_t *torso_new = make_card(main, 424, 388, 368, 70);
    lv_obj_set_style_shadow_width(torso_new, 0, LV_PART_MAIN);
    make_text(torso_new, "躯干", 0, 0, 76, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_corr_torso = make_text(torso_new, "躯干 0 | 前倾 0° | 抬胸收紧核心", 84, 4, 250, font_14(), 0x314551, LV_LABEL_LONG_DOT);

    lv_obj_t *balance_new = make_card(main, 24, 476, 376, 70);
    lv_obj_set_style_shadow_width(balance_new, 0, LV_PART_MAIN);
    make_text(balance_new, "平衡", 0, 0, 76, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_corr_balance = make_text(balance_new, "左右平衡 0 | 膝角差 0°", 84, 4, 260, font_14(), 0x314551, LV_LABEL_LONG_DOT);

    lv_obj_t *track_new = make_card(main, 424, 476, 368, 70);
    lv_obj_set_style_shadow_width(track_new, 0, LV_PART_MAIN);
    make_text(track_new, "识别", 0, 0, 76, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_corr_track = make_text(track_new, "识别稳定 0 | 有效点 0", 84, 4, 250, font_14(), 0x314551, LV_LABEL_LONG_DOT);

    update_correction_ui_from_snapshot(training_get_snapshot());
    return;

    lv_obj_t *analysis = make_card(main, 24, 98, 376, 210);
    make_text(analysis, "当前分析", 0, 0, 200, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(analysis, "膝盖轨迹", 0, 46, 140, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_progress(analysis, 142, 52, 190, 82, 0x22a06b);
    make_text(analysis, "髋部深度", 0, 84, 140, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_progress(analysis, 142, 90, 190, 68, 0xd9822b);
    make_text(analysis, "躯干前倾", 0, 122, 140, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_progress(analysis, 142, 128, 190, 74, 0x2f7dd1);
    make_text(analysis, "建议: 髋部再低一点，起身阶段放慢速度。", 0, 164, 320, font_14(), 0x687783);

    lv_obj_t *tech = make_card(main, 424, 98, 368, 210);
    make_text(tech, "技术视图", 0, 0, 190, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(tech, "PC路径: JPEG图像 -> MoveNet -> 平滑骨架。", 0, 44, 320, font_14(), 0x314551);
    make_text(tech, "本地路径: 板端模型用于离线对比。", 0, 84, 320, font_14(), 0x314551);
    make_text(tech, "绿色和红色骨架用于区分PC与本地结果。", 0, 124, 320, font_14(), 0x687783);

    const char *items[] = {"膝内扣", "深度不足", "躯干前倾", "左右节奏不一致"};
    for (int i = 0; i < 4; i++) {
        lv_obj_t *row = make_card(main, 24 + (i % 2) * 392, 332 + (i / 2) * 96, 376, 72);
        if (!row) {
            continue;
        }
        lv_obj_set_style_shadow_width(row, 0, LV_PART_MAIN);
        make_text(row, items[i], 0, 0, 260, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
        make_text(row, "错误出现时仅显示简短提示。", 0, 34, 320, font_14(), 0x687783);
    }
}

static void create_courses_page(void)
{
    scr_courses = lv_obj_create(NULL);
    if (!scr_courses) {
        mark_lvgl_build_failed("courses screen");
        return;
    }
    lv_obj_t *main = make_product_shell(scr_courses,
                                        "courses",
                                        "课程库",
                                        "课程组合多个动作；单项训练保留在选择训练页。");
    if (!main) {
        return;
    }
    add_top_status(main, "SD课程", "MJPEG播放", "无音频");

    lv_obj_t *list_card = make_card(main, 24, 98, 320, 448);
    make_text(list_card, "课程视频", 0, 0, 160, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_product_button(list_card, 196, 0, 92, 32, "重新扫描", course_video_refresh_cb, false);
    s_course_list = lv_obj_create(list_card);
    if (s_course_list) {
        lv_obj_remove_style_all(s_course_list);
        lv_obj_set_pos(s_course_list, 0, 48);
        lv_obj_set_size(s_course_list, 292, 360);
        lv_obj_set_scroll_dir(s_course_list, LV_DIR_VER);
        lv_obj_set_style_pad_all(s_course_list, 0, LV_PART_MAIN);
    }

    lv_obj_t *player_card = make_card(main, 368, 98, 424, 330);
    s_course_player_card = player_card;
    make_text(player_card, "课程播放", 0, 0, 110, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_product_button(player_card, 152, 0, 68, 32, "暂停", video_player_pause_toggle_cb, false);
    make_product_button(player_card, 228, 0, 58, 32, "停止", video_player_stop_button_cb, false, s_course_player_host);
    s_course_zoom_button = make_product_button(player_card, 314, 0, 70, 32, "放大", course_video_zoom_cb, false);
    s_course_title_label = make_text(player_card, "请选择课程视频", 0, 36, 384, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    s_course_player_host = lv_obj_create(player_card);
    if (s_course_player_host) {
        lv_obj_set_pos(s_course_player_host, 0, 56);
        lv_obj_set_size(s_course_player_host, 384, 216);
        lv_obj_clear_flag(s_course_player_host, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_set_style_bg_color(s_course_player_host, lv_color_black(), LV_PART_MAIN);
        lv_obj_set_style_bg_opa(s_course_player_host, LV_OPA_COVER, LV_PART_MAIN);
        lv_obj_set_style_border_width(s_course_player_host, 0, LV_PART_MAIN);
        lv_obj_set_style_pad_all(s_course_player_host, 0, LV_PART_MAIN);
        lv_obj_set_style_radius(s_course_player_host, 8, LV_PART_MAIN);
        lv_obj_set_style_clip_corner(s_course_player_host, true, LV_PART_MAIN);
        make_text(s_course_player_host, "选择左侧课程后播放", 112, 92, 180, font_14(), 0xd7dee8, LV_LABEL_LONG_DOT);
    }
    s_course_status_label = make_text(player_card,
                                      "请把 sdcard_package/FIT_VIDEO 复制到 SD 卡根目录。",
                                      0,
                                      286,
                                      380,
                                      font_14(),
                                      0x314551,
                                      LV_LABEL_LONG_WRAP);

    lv_obj_t *info = make_card(main, 368, 450, 424, 96);
    make_text(info, "SD 卡目录", 0, 0, 160, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(info,
              "/FIT_VIDEO/courses/manifest.csv\n视频格式：MJPEG，无音频，英文文件名。",
              0,
              34,
              376,
              font_14(),
              0x687783,
              LV_LABEL_LONG_WRAP);
    course_render_list();
    apply_course_video_zoom(false);
    return;

    const char *names[] = {"基础入门", "推力训练", "拉力训练", "力量基础"};
    const char *desc[] = {
        "深蹲、俯卧撑和平板支撑入门。",
        "俯卧撑和卧推，强化胸肩三头控制。",
        "引体向上和硬拉，强化背部与髋部发力。",
        "深蹲、硬拉、平板支撑组成基础力量课。",
    };
    const char *tags[] = {"初级 / 20分钟", "推力 / 28分钟", "拉力 / 24分钟", "力量 / 30分钟"};
    for (int i = 0; i < 4; i++) {
        const int32_t x = 24 + (i % 2) * 392;
        const int32_t y = 102 + (i / 2) * 170;
        lv_obj_t *card = make_card(main, x, y, 368, 144);
        make_text(card, names[i], 0, 0, 210, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
        make_text(card, desc[i], 0, 36, 300, font_14(), 0x687783);
        make_text(card, tags[i], 0, 94, 210, font_14(), 0x145883, LV_LABEL_LONG_DOT);
        make_product_button(card, 244, 88, 92, 34, "打开", show_start_cb, false);
    }

    lv_obj_t *comp = make_card(main, 24, 454, 768, 90);
    make_text(comp, "课程组成", 0, 0, 250, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(comp, "热身 -> 动作练习 -> 实时纠正 -> 放松总结。", 0, 38, 690, font_14(), 0x687783);
}

static void create_plan_page(void)
{
    scr_plan = lv_obj_create(NULL);
    if (!scr_plan) {
        mark_lvgl_build_failed("plan screen");
        return;
    }
    lv_obj_t *main = make_product_shell(scr_plan,
                                        "plan",
                                        "训练计划",
                                        "周训练安排和今日任务清晰分开。");
    if (!main) {
        return;
    }
    add_top_status(main, "今日任务", "本周计划", "可自定");

    const int rec_index = user_recommended_training_index();
    const training_profile_t *today_profile = training_profile_by_index(rec_index);
    const int today_items[] = {rec_index, 5};
    const char *labels[] = {"主训练", "核心训练"};
    char target_text[48];
    training_target_text_by_index(rec_index, target_text, sizeof(target_text));
    int today_percent = 0;
    for (int i = 0; i < 2; ++i) {
        today_percent += training_record_progress_for_index(today_items[i]);
    }
    today_percent /= 2;

    lv_obj_t *switch_btn = make_product_button(main, 666, 64, 118, 32, "查看本周", plan_toggle_week_cb, false);
    if (switch_btn) {
        btn_plan_week_label = lv_obj_get_child(switch_btn, 0);
    }

    panel_plan_today = lv_obj_create(main);
    if (!panel_plan_today) {
        mark_lvgl_build_failed("plan today panel");
        return;
    }
    lv_obj_remove_style_all(panel_plan_today);
    lv_obj_set_pos(panel_plan_today, 0, 104);
    lv_obj_set_size(panel_plan_today, 816, 454);
    lv_obj_clear_flag(panel_plan_today, LV_OBJ_FLAG_SCROLLABLE);

    lv_obj_t *today = make_card(panel_plan_today, 24, 0, 500, 200);
    make_text(today, "今日训练", 0, 0, 180, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    char line[120];
    snprintf(line, sizeof(line), "%s  ·  %s", today_profile->name, target_text);
    make_text(today, line, 0, 38, 430, font_16(), 0x145883, LV_LABEL_LONG_DOT);
    make_text(today, today_profile->focus, 0, 70, 430, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_text(today, "今日完成度", 0, 108, 120, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    char percent_text[32];
    snprintf(percent_text, sizeof(percent_text), "%d%%", today_percent);
    make_text(today, percent_text, 378, 106, 70, font_18(), 0x22a06b, LV_LABEL_LONG_DOT);
    make_progress(today, 0, 138, 430, today_percent, 0x22a06b);
    make_product_button(today, 0, 152, 112, 34, "开始训练", show_start_cb, true);
    make_product_button(today, 126, 152, 112, 34, "制定计划", show_start_cb, false);

    lv_obj_t *summary = make_card(panel_plan_today, 544, 0, 248, 200);
    make_text(summary, "计划状态", 0, 0, 150, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(summary, "根据基础信息和历史训练给出今日推荐。", 0, 38, 204, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
    char sessions_text[32];
    snprintf(sessions_text, sizeof(sessions_text), "%u次", (unsigned)s_training_record.sessions);
    make_metric(summary, 0, 92, 94, 78, "训练", sessions_text, s_training_record.sessions > 0 ? 76 : 12, 0x145883);
    char score_text[32];
    snprintf(score_text, sizeof(score_text), s_training_record.sessions > 0 ? "%d分" : "--", s_training_record.last_score);
    make_metric(summary, 110, 92, 94, 78, "得分", score_text, s_training_record.last_score, 0x22a06b);

    lv_obj_t *today_list = make_card(panel_plan_today, 24, 226, 768, 170);
    make_text(today_list, "今日动作安排", 0, 0, 180, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    for (int i = 0; i < 2; ++i) {
        const training_profile_t *p = training_profile_by_index(today_items[i]);
        char item_target[48];
        training_target_text_by_index(today_items[i], item_target, sizeof(item_target));
        const int y = 46 + i * 52;
        make_dot(today_list, 0, y + 8, p->color);
        make_text(today_list, labels[i], 18, y, 72, font_14(), 0x687783, LV_LABEL_LONG_DOT);
        make_text(today_list, p->name, 104, y, 100, font_14(), 0x16232d, LV_LABEL_LONG_DOT);
        make_text(today_list, item_target, 228, y, 120, font_14(), 0x314551, LV_LABEL_LONG_DOT);
        make_text(today_list, p->focus, 374, y, 330, font_14(), 0x687783, LV_LABEL_LONG_DOT);
        const int item_progress = training_record_progress_for_index(today_items[i]);
        make_progress(today_list, 104, y + 28, 244, item_progress, p->color);
    }

    panel_plan_week = lv_obj_create(main);
    if (!panel_plan_week) {
        mark_lvgl_build_failed("plan week panel");
        return;
    }
    lv_obj_remove_style_all(panel_plan_week);
    lv_obj_set_pos(panel_plan_week, 0, 104);
    lv_obj_set_size(panel_plan_week, 816, 454);
    lv_obj_clear_flag(panel_plan_week, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_add_flag(panel_plan_week, LV_OBJ_FLAG_HIDDEN);

    lv_obj_t *week = make_card(panel_plan_week, 24, 0, 768, 270);
    make_text(week, "本周训练计划", 0, 0, 220, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(week, "每天一个主训练，搭配核心或恢复；目标可在选择训练页调整。", 0, 34, 620, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    const char *days[] = {"周一", "周二", "周三", "周四", "周五", "周六", "周日"};
    const int plan_index[] = {0, 2, -1, 3, 4, 1, 5};
    const int today_weekday = current_weekday_monday_index();
    for (int i = 0; i < 7; i++) {
        const int32_t x = 0 + i * 104;
        lv_obj_t *card = make_card(week, x, 76, 92, 144);
        if (!card) {
            continue;
        }
        lv_obj_set_style_shadow_width(card, 0, LV_PART_MAIN);
        const bool is_today = i == today_weekday;
        if (is_today) {
            lv_obj_set_style_border_width(card, 2, LV_PART_MAIN);
            lv_obj_set_style_border_color(card, lv_color_hex(0x145883), LV_PART_MAIN);
            lv_obj_set_style_bg_color(card, lv_color_hex(0xf0f7fa), LV_PART_MAIN);
        }
        make_text(card, days[i], 0, 0, 64, font_16(), is_today ? 0x145883 : 0x16232d, LV_LABEL_LONG_DOT);
        if (is_today) {
            make_text(card, "今天", 48, 1, 36, font_14(), 0x145883, LV_LABEL_LONG_DOT);
        }
        if (plan_index[i] >= 0) {
            const training_profile_t *p = training_profile_by_index(plan_index[i]);
            make_text(card, p->name, 0, 36, 64, font_14(), 0x314551, LV_LABEL_LONG_DOT);
            char mini_target[32];
            training_target_text_by_index(plan_index[i], mini_target, sizeof(mini_target));
            make_text(card, mini_target, 0, 62, 64, font_14(), 0x687783, LV_LABEL_LONG_DOT);
            const int progress = is_today ? today_percent : 0;
            make_progress(card, 0, 104, 64, progress, p->color);
        } else {
            make_text(card, "休息", 0, 44, 64, font_14(), 0x687783, LV_LABEL_LONG_DOT);
            make_text(card, "拉伸恢复", 0, 70, 64, font_14(), 0x687783, LV_LABEL_LONG_DOT);
            make_progress(card, 0, 104, 64, is_today ? 100 : 0, 0x9fb2bf);
        }
    }

    lv_obj_t *custom = make_card(panel_plan_week, 24, 296, 768, 112);
    make_text(custom, "制定计划", 0, 0, 160, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_text(custom, "进入选择训练页后，可以分别调整每个动作的组数、次数或保持秒数。", 0, 36, 560, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_product_button(custom, 600, 26, 112, 38, "去制定", show_start_cb, true);
}

static void create_music_page(void)
{
    scr_music = lv_obj_create(NULL);
    if (!scr_music) {
        mark_lvgl_build_failed("music screen");
        return;
    }
    lv_obj_t *main = make_product_shell(scr_music,
                                        "music",
                                        "音乐播放",
                                        "训练时后台播放本地音乐，退出页面也会继续。");
    if (!main) {
        return;
    }
    add_top_status(main, "SD卡", "WAV播放", "后台运行");

    std::vector<music_track_t> tracks;
    music_lock();
    tracks = s_music_tracks;
    music_unlock();

    lv_obj_t *player = make_card(main, 24, 94, 506, 420);
    if (!player) {
        return;
    }
    lv_obj_set_style_bg_color(player, lv_color_hex(0xffffff), LV_PART_MAIN);

    lv_obj_t *cover = lv_obj_create(player);
    if (cover) {
        lv_obj_set_pos(cover, 0, 0);
        lv_obj_set_size(cover, 214, 214);
        lv_obj_clear_flag(cover, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_set_style_bg_color(cover, lv_color_hex(0x0b344f), LV_PART_MAIN);
        lv_obj_set_style_border_width(cover, 0, LV_PART_MAIN);
        lv_obj_set_style_radius(cover, 8, LV_PART_MAIN);
        lv_obj_set_style_pad_all(cover, 0, LV_PART_MAIN);

        img_music_cover = lv_image_create(cover);
        if (img_music_cover) {
            lv_obj_set_size(img_music_cover, 214, 214);
            lv_obj_center(img_music_cover);
            lv_obj_add_flag(img_music_cover, LV_OBJ_FLAG_HIDDEN);
        }
        make_text(cover, "FIT MIRROR", 22, 26, 160, font_14(), 0x9fb2bf, LV_LABEL_LONG_DOT);
        lbl_music_cover_title = make_text(cover, "本地音乐", 22, 78, 160, font_18(), 0xffffff, LV_LABEL_LONG_WRAP);
        make_text(cover, "SD CARD", 22, 154, 160, font_14(), 0x9fb2bf, LV_LABEL_LONG_DOT);
    } else {
        mark_lvgl_build_failed("music cover");
    }

    lbl_music_title = make_text(player, "--", 244, 12, 210, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_music_artist = make_text(player, "本地音乐", 244, 44, 210, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    lbl_music_status = make_text(player, "等待SD卡", 244, 82, 210, font_14(), 0x22a06b, LV_LABEL_LONG_DOT);
    lbl_music_time = make_text(player, "00:00 / 00:00", 244, 124, 210, font_14(), 0x314551, LV_LABEL_LONG_DOT);

    bar_music_progress = lv_bar_create(player);
    if (bar_music_progress) {
        lv_obj_set_pos(bar_music_progress, 244, 156);
        lv_obj_set_size(bar_music_progress, 220, 10);
        lv_bar_set_range(bar_music_progress, 0, 1000);
        lv_obj_set_style_radius(bar_music_progress, 8, LV_PART_MAIN);
        lv_obj_set_style_radius(bar_music_progress, 8, LV_PART_INDICATOR);
        lv_obj_set_style_bg_color(bar_music_progress, lv_color_hex(0xe4edf1), LV_PART_MAIN);
        lv_obj_set_style_bg_color(bar_music_progress, lv_color_hex(0x145883), LV_PART_INDICATOR);
    } else {
        mark_lvgl_build_failed("music progress");
    }

    lv_obj_t *prev = make_product_button(player, 244, 194, 58, 42, "上一", music_prev_cb, false);
    (void)prev;
    lv_obj_t *play = make_product_button(player, 314, 194, 76, 42, "播放", music_play_toggle_cb, true);
    if (play) {
        btn_music_play_label = lv_obj_get_child(play, 0);
    }
    lv_obj_t *next = make_product_button(player, 402, 194, 58, 42, "下一", music_next_cb, false);
    (void)next;

    make_product_button(player, 244, 252, 92, 34, "停止", music_stop_cb, false);
    make_product_button(player, 348, 252, 112, 34, "重新扫描", music_rescan_cb, false);

    make_text(player, "音量", 0, 252, 60, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    slider_music_volume = lv_slider_create(player);
    if (slider_music_volume) {
        lv_obj_set_pos(slider_music_volume, 58, 258);
        lv_obj_set_size(slider_music_volume, 154, 12);
        lv_slider_set_range(slider_music_volume, 0, 100);
        lv_slider_set_value(slider_music_volume, MUSIC_DEFAULT_VOLUME, LV_ANIM_OFF);
        lv_obj_set_style_bg_color(slider_music_volume, lv_color_hex(0xe4edf1), LV_PART_MAIN);
        lv_obj_set_style_bg_color(slider_music_volume, lv_color_hex(0x145883), LV_PART_INDICATOR);
        lv_obj_set_style_bg_color(slider_music_volume, lv_color_hex(0x145883), LV_PART_KNOB);
        lv_obj_add_event_cb(slider_music_volume, music_volume_cb, LV_EVENT_VALUE_CHANGED, NULL);
    } else {
        mark_lvgl_build_failed("music volume");
    }

    make_text(player, "曲目信息", 0, 310, 90, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    lbl_music_cover_path = make_text(player, "本地音乐 · TF卡曲库", 0, 340, 452, font_14(), 0x314551, LV_LABEL_LONG_DOT);

    lv_obj_t *queue = make_card(main, 552, 94, 240, 420);
    if (queue) {
        make_text(queue, "播放列表", 0, 0, 160, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    }
    if (queue && tracks.empty()) {
        make_text(queue, "没有读取到曲目，请检查 /MUSIC/playlist.txt", 0, 48, 196, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
    } else if (queue) {
        const int max_rows = tracks.size() > 5 ? 5 : (int)tracks.size();
        for (int i = 0; i < max_rows; i++) {
            lv_obj_t *row = lv_obj_create(queue);
            if (!row) {
                mark_lvgl_build_failed("music row");
                continue;
            }
            lv_obj_set_pos(row, 0, 44 + i * 64);
            lv_obj_set_size(row, 196, 54);
            lv_obj_clear_flag(row, LV_OBJ_FLAG_SCROLLABLE);
            lv_obj_add_flag(row, LV_OBJ_FLAG_CLICKABLE);
            lv_obj_set_style_bg_color(row, lv_color_hex(0xf7fafb), LV_PART_MAIN);
            lv_obj_set_style_border_color(row, lv_color_hex(0xd9e2e8), LV_PART_MAIN);
            lv_obj_set_style_border_width(row, 1, LV_PART_MAIN);
            lv_obj_set_style_radius(row, 8, LV_PART_MAIN);
            lv_obj_set_style_pad_all(row, 0, LV_PART_MAIN);
            lv_obj_add_event_cb(row, music_track_cb, LV_EVENT_CLICKED, (void *)(intptr_t)i);
            make_text(row, tracks[i].title.c_str(), 14, 8, 150, font_14(), 0x16232d, LV_LABEL_LONG_DOT);
            make_text(row, "本地音乐", 14, 30, 150, font_14(), 0x687783, LV_LABEL_LONG_DOT);
        }
    }

    lv_obj_t *hint = make_card(main, 24, 532, 768, 46);
    if (hint) {
        lv_obj_set_style_shadow_width(hint, 0, LV_PART_MAIN);
        lv_obj_set_style_bg_color(hint, lv_color_hex(0xeaf7f1), LV_PART_MAIN);
        make_text(hint, "提示: 退出音乐页后会继续播放；训练页右侧可一键停止背景音乐。", 0, 0, 720, font_14(), 0x1a7f53, LV_LABEL_LONG_DOT);
    }
    update_music_ui();
}

static void create_report_page(void)
{
    scr_report = lv_obj_create(NULL);
    if (!scr_report) {
        mark_lvgl_build_failed("report screen");
        return;
    }
    lv_obj_t *main = make_product_shell(scr_report,
                                        "report",
                                        "训练报告",
                                        "训练后的总结、趋势和下一步建议。");
    if (!main) {
        return;
    }
    add_top_status(main, "NVS记录", "按日期查看", "在线+离线");

    char today[16];
    const bool has_today = training_history_date_prefix(today, sizeof(today));
    int today_sessions = 0;
    int today_score_sum = 0;
    int today_count_sum = 0;
    uint32_t today_elapsed_ms = 0;
    for (int i = 0; i < s_training_history_count; ++i) {
        const training_history_item_t &item = s_training_history[i];
        if (!item.valid || !has_today || strncmp(item.time, today, strlen(today)) != 0) {
            continue;
        }
        today_sessions++;
        today_score_sum += std::max(0, item.score);
        today_count_sum += std::max(0, item.count);
        today_elapsed_ms += item.elapsed_ms;
    }

    if (today_sessions == 0 && s_training_record.sessions > 0 && training_record_is_today()) {
        today_sessions = 1;
        today_score_sum = std::max(0, s_training_record.last_score);
        today_count_sum = std::max(0, s_training_record.last_count);
        today_elapsed_ms = s_training_record.last_elapsed_ms;
    }

    char today_records[24];
    char today_score[24];
    char today_count[24];
    char history_count[24];
    snprintf(today_records, sizeof(today_records), "%d条", today_sessions);
    snprintf(today_score,
             sizeof(today_score),
             today_sessions > 0 ? "%d分" : "--",
             today_sessions > 0 ? today_score_sum / today_sessions : 0);
    snprintf(today_count, sizeof(today_count), "%d次", today_count_sum);
    snprintf(history_count,
             sizeof(history_count),
             "%d条",
             std::max(s_training_history_count, (int)s_training_record.sessions));

    make_metric(main, 24, 104, 176, 102, "今日记录", today_records, today_sessions > 0 ? 78 : 12, 0x145883);
    make_metric(main, 216, 104, 176, 102, "今日均分", today_score, today_sessions > 0 ? today_score_sum / today_sessions : 0, 0x22a06b);
    make_metric(main, 408, 104, 176, 102, "今日次数", today_count, today_count_sum > 0 ? 82 : 10, 0x2f7dd1);
    char today_elapsed[24];
    format_mmss(today_elapsed_ms, today_elapsed, sizeof(today_elapsed));
    make_metric(main, 600, 104, 192, 102, "累计用时", today_sessions > 0 ? today_elapsed : "--", 70, 0xd9822b);

    lv_obj_t *history = make_card(main, 24, 226, 500, 330);
    if (history) {
        make_text(history, "训练记录", 0, 0, 160, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
        char subtitle[80];
        snprintf(subtitle, sizeof(subtitle), "今日 %s · 最近 %s", today_records, history_count);
        make_text(history, subtitle, 170, 3, 270, font_14(), 0x687783, LV_LABEL_LONG_DOT);

        lv_obj_t *scroll = lv_obj_create(history);
        if (scroll) {
            lv_obj_set_pos(scroll, 0, 42);
            lv_obj_set_size(scroll, 456, 248);
            lv_obj_set_style_bg_opa(scroll, LV_OPA_TRANSP, LV_PART_MAIN);
            lv_obj_set_style_border_width(scroll, 0, LV_PART_MAIN);
            lv_obj_set_style_pad_all(scroll, 0, LV_PART_MAIN);
            lv_obj_set_style_pad_right(scroll, 6, LV_PART_MAIN);
            lv_obj_set_scroll_dir(scroll, LV_DIR_VER);
            char lines[1280];
            size_t off = 0;
            auto append_line = [&](const char *fmt, ...) {
                if (off >= sizeof(lines)) {
                    return;
                }
                va_list ap;
                va_start(ap, fmt);
                const int n = vsnprintf(lines + off, sizeof(lines) - off, fmt, ap);
                va_end(ap);
                if (n > 0) {
                    off += std::min<size_t>((size_t)n, sizeof(lines) - off - 1);
                }
            };
            lines[0] = '\0';
            if (s_training_history_count > 0) {
                char last_date[16] = {};
                for (int i = 0; i < s_training_history_count; ++i) {
                    const training_history_item_t &item = s_training_history[i];
                    if (!item.valid) {
                        continue;
                    }
                    char date[16] = "--";
                    if (strlen(item.time) >= 10) {
                        memcpy(date, item.time, 10);
                        date[10] = '\0';
                    }
                    if (strcmp(date, last_date) != 0) {
                        append_line("%s%s\n", i == 0 ? "" : "\n", date);
                        strlcpy(last_date, date, sizeof(last_date));
                    }
                    char elapsed[16];
                    format_mmss(item.elapsed_ms, elapsed, sizeof(elapsed));
                    append_line("  %s  %s  %d次  %d分  %s\n",
                                training_history_source_text(item.source),
                                item.action,
                                item.count,
                                item.score,
                                elapsed);
                    if (item.tip[0]) {
                        append_line("  纠错: %.54s\n", item.tip);
                    }
                    if (item.reps[0]) {
                        append_line("  单次: %.62s\n", item.reps);
                    }
                }
            } else if (s_training_record.sessions > 0) {
                append_line("%s\n  实时  %s  %d/%d  %d分\n  纠错: %s\n",
                            s_training_record.last_time,
                            s_training_record.last_training,
                            s_training_record.last_count,
                            s_training_record.last_target,
                            s_training_record.last_score,
                            s_training_record.last_tip);
            } else {
                append_line("暂无训练记录。\n完成实时训练或离线检测后会自动保存到这里。");
            }
            lv_obj_t *text = make_text(scroll, lines, 0, 0, 430, font_14(), 0x314551, LV_LABEL_LONG_WRAP);
            if (text) {
                lv_obj_set_width(text, 430);
            }
        }
    }

    const training_history_item_t *last = s_training_history_count > 0 ? &s_training_history[0] : nullptr;
    lv_obj_t *detail = make_card(main, 540, 226, 252, 330);
    if (detail) {
        make_text(detail, "最近一次", 0, 0, 180, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
        char line[160];
        if (last && last->valid) {
            snprintf(line,
                     sizeof(line),
                     "%s · %s\n%s",
                     training_history_source_text(last->source),
                     last->action,
                     last->time);
            make_text(detail, line, 0, 38, 210, font_14(), 0x314551, LV_LABEL_LONG_WRAP);
            snprintf(line,
                     sizeof(line),
                     "次数 %d  总分 %d  均分 %d\n幅度%d 膝%d 髋%d 躯干%d\n稳定%d 跟踪%d",
                     last->count,
                     last->score,
                     last->rep_score > 0 ? last->rep_score : last->score,
                     last->score_depth,
                     last->score_knee,
                     last->score_hip,
                     last->score_torso,
                     last->score_balance,
                     last->score_track);
            make_text(detail, line, 0, 100, 210, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
            make_text(detail, last->tip[0] ? last->tip : "保持动作完整和稳定。", 0, 184, 210, font_14(), 0x314551, LV_LABEL_LONG_WRAP);
        } else if (s_training_record.sessions > 0) {
            snprintf(line,
                     sizeof(line),
                     "实时 · %s\n%s",
                     s_training_record.last_training,
                     s_training_record.last_time);
            make_text(detail, line, 0, 38, 210, font_14(), 0x314551, LV_LABEL_LONG_WRAP);
            snprintf(line,
                     sizeof(line),
                     "次数 %d/%d  总分 %d\n幅度%d 膝%d 髋%d\n躯干%d 稳定%d 跟踪%d",
                     s_training_record.last_count,
                     s_training_record.last_target,
                     s_training_record.last_score,
                     s_training_record.last_score_depth,
                     s_training_record.last_score_knee,
                     s_training_record.last_score_hip,
                     s_training_record.last_score_torso,
                     s_training_record.last_score_balance,
                     s_training_record.last_score_track);
            make_text(detail, line, 0, 100, 210, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
            make_text(detail, s_training_record.last_tip, 0, 184, 210, font_14(), 0x314551, LV_LABEL_LONG_WRAP);
        } else {
            make_text(detail, "还没有训练数据。\n先完成一次实时训练或离线检测。", 0, 48, 210, font_14(), 0x687783, LV_LABEL_LONG_WRAP);
        }
        make_product_button(detail, 0, 260, 96, 34, "去训练", show_start_cb, true);
        make_product_button(detail, 108, 260, 96, 34, "训练计划", show_plan_cb, false);
    }
}

static void create_settings_page(void)
{
    scr_settings = lv_obj_create(NULL);
    if (!scr_settings) {
        mark_lvgl_build_failed("settings screen");
        return;
    }
    lv_obj_t *main = make_product_shell(scr_settings,
                                        "settings",
                                        "设备设置",
                                        "网络、PC服务、推理模式和显示选项。");
    if (!main) {
        return;
    }
    add_top_status(main, "ESP32-P4", "在线", "配置");

    lv_obj_t *wifi = make_card(main, 24, 100, 376, 250);
    make_text(wifi, "Wi-Fi连接", 0, 0, 210, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_wifi_state = make_text(wifi, "状态: --", 214, 4, 128, font_14(), wifi_ui_state_color(), LV_LABEL_LONG_DOT);
    lbl_wifi_saved_ssid = make_text(wifi, "当前 SSID: --", 0, 42, 328, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    lbl_wifi_ip = make_text(wifi, "IP: --", 0, 70, 328, font_14(), 0x687783, LV_LABEL_LONG_DOT);

    make_text(wifi, "SSID", 0, 104, 80, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    ta_wifi_ssid = lv_textarea_create(wifi);
    lv_obj_set_size(ta_wifi_ssid, 214, 38);
    lv_obj_set_pos(ta_wifi_ssid, 90, 98);
    lv_textarea_set_one_line(ta_wifi_ssid, true);
    lv_textarea_set_max_length(ta_wifi_ssid, 32);
    lv_textarea_set_placeholder_text(ta_wifi_ssid, "SSID");
    lv_textarea_set_text(ta_wifi_ssid, s_wifi_ssid);
    style_textarea(ta_wifi_ssid);

    make_text(wifi, "密码", 0, 150, 80, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    ta_wifi_pass = lv_textarea_create(wifi);
    lv_obj_set_size(ta_wifi_pass, 214, 38);
    lv_obj_set_pos(ta_wifi_pass, 90, 144);
    lv_textarea_set_one_line(ta_wifi_pass, true);
    lv_textarea_set_password_mode(ta_wifi_pass, true);
    lv_textarea_set_password_bullet(ta_wifi_pass, "*");
    lv_textarea_set_max_length(ta_wifi_pass, 64);
    lv_textarea_set_placeholder_text(ta_wifi_pass, "Password");
    lv_textarea_set_text(ta_wifi_pass, s_wifi_password);
    style_textarea(ta_wifi_pass);

    make_product_button(wifi, 0, 196, 82, 34, "连接", wifi_connect_cb, true);
    make_product_button(wifi, 92, 196, 82, 34, "重连", wifi_reconnect_cb, false);
    make_product_button(wifi, 184, 196, 82, 34, "断开", wifi_disconnect_cb, false);
    make_product_button(wifi, 276, 196, 82, 34, "默认", wifi_restore_default_cb, false);
    update_wifi_labels_ui();

    lv_obj_t *pc = make_card(main, 424, 100, 368, 250);
    make_text(pc, "PC姿态服务", 0, 0, 220, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    make_pill(pc, 230, 0, 94, s_pc_pose_wifi_ready.load() ? "在线" : "待连接", s_pc_pose_wifi_ready.load() ? 0x22a06b : 0xd9822b);
    make_text(pc, "地址", 0, 52, 80, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_text(pc, "192.168.43.131", 90, 52, 220, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_text(pc, "端口", 0, 86, 80, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    make_text(pc, "8000", 90, 86, 220, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_text(pc, "模式: PC优先，本地对比", 0, 132, 300, font_14(), 0x145883, LV_LABEL_LONG_DOT);
    make_text(pc, "Wi-Fi连通后，实时训练页会推送JPEG到PC端。", 0, 170, 300, font_14(), 0x687783);
    lbl_settings_pc = make_text(pc, "PC服务: --", 0, 212, 320, font_14(), 0x314551, LV_LABEL_LONG_DOT);

    lv_obj_t *account = make_card(main, 24, 374, 376, 172);
    make_text(account, "账号与人脸", 0, 0, 200, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_settings_account = make_text(account, "账号: --", 0, 40, 330, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    lbl_settings_face = make_text(account, "人脸库: --", 0, 68, 330, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    lbl_settings_profile = make_text(account, "资料: --", 0, 96, 330, font_14(), 0x145883, LV_LABEL_LONG_DOT);
    make_product_button(account, 0, 126, 78, 32, "锁定", auth_logout_cb, true);
    make_product_button(account, 88, 126, 78, 32, "资料", profile_edit_cb, false);
    make_product_button(account, 176, 126, 78, 32, "清账号", auth_clear_account_cb, false);
    make_product_button(account, 264, 126, 78, 32, "清人脸", face_clear_storage_cb, false);

    lv_obj_t *display = make_card(main, 424, 374, 368, 172);
    make_text(display, "系统状态", 0, 0, 180, font_18(), 0x16232d, LV_LABEL_LONG_DOT);
    lbl_settings_heap = make_text(display, "内存: --", 0, 40, 330, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    lbl_settings_uptime = make_text(display, "运行: --", 0, 68, 330, font_14(), 0x314551, LV_LABEL_LONG_DOT);
    make_text(display, "推理: PC优先 / MoveNet备份", 0, 96, 330, font_14(), 0x145883, LV_LABEL_LONG_DOT);
    make_text(display, "数据: NVS保存账号、Wi-Fi、资料和记录", 0, 124, 330, font_14(), 0x687783, LV_LABEL_LONG_DOT);
    update_settings_status_ui();

    kb_settings = lv_keyboard_create(scr_settings);
    lv_obj_set_size(kb_settings, SCREEN_W, 230);
    lv_obj_align(kb_settings, LV_ALIGN_BOTTOM_MID, 0, 0);
    lv_obj_add_flag(kb_settings, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_event_cb(ta_wifi_ssid, shared_keyboard_cb, LV_EVENT_ALL, kb_settings);
    lv_obj_add_event_cb(ta_wifi_pass, shared_keyboard_cb, LV_EVENT_ALL, kb_settings);
    lv_obj_add_event_cb(kb_settings, shared_keyboard_cb, LV_EVENT_READY, kb_settings);
    lv_obj_add_event_cb(kb_settings, shared_keyboard_cb, LV_EVENT_CANCEL, kb_settings);
}

static void create_face_page(void)
{
    scr_face = lv_obj_create(NULL);
    style_screen(scr_face);

    view_face = make_preview(scr_face);
    cv_face = lv_canvas_create(view_face);
    lv_obj_clear_flag(cv_face, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_center(cv_face);

    box_face = lv_obj_create(scr_face);
    lv_obj_remove_style_all(box_face);
    lv_obj_clear_flag(box_face, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_set_style_bg_opa(box_face, LV_OPA_TRANSP, LV_PART_MAIN);
    lv_obj_set_style_border_color(box_face, lv_color_hex(0x00ff44), LV_PART_MAIN);
    lv_obj_set_style_border_width(box_face, 4, LV_PART_MAIN);
    lv_obj_add_flag(box_face, LV_OBJ_FLAG_HIDDEN);

    lbl_face_name = lv_label_create(scr_face);
    lv_obj_clear_flag(lbl_face_name, LV_OBJ_FLAG_CLICKABLE);
    set_label(lbl_face_name, "人脸");
    lv_obj_set_style_bg_color(lbl_face_name, lv_color_hex(0x00aa33), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(lbl_face_name, LV_OPA_80, LV_PART_MAIN);
    lv_obj_set_style_text_color(lbl_face_name, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_pad_hor(lbl_face_name, 8, LV_PART_MAIN);
    lv_obj_set_style_pad_ver(lbl_face_name, 4, LV_PART_MAIN);
    lv_obj_set_style_radius(lbl_face_name, 6, LV_PART_MAIN);
    lv_obj_add_flag(lbl_face_name, LV_OBJ_FLAG_HIDDEN);

    lv_obj_t *panel = make_side_panel(scr_face);

    lv_obj_t *title = lv_label_create(panel);
    set_label(title, "人脸");
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 4);

    lv_obj_t *back = make_button(panel, SIDE_PANEL_W - 24, 48, "返回");
    lv_obj_align(back, LV_ALIGN_TOP_MID, 0, 40);
    lv_obj_add_event_cb(back, show_menu_cb, LV_EVENT_CLICKED, NULL);

    lv_obj_t *name_label = lv_label_create(panel);
    set_label(name_label, "姓名");
    lv_obj_align(name_label, LV_ALIGN_TOP_LEFT, 0, 100);

    ta_name = lv_textarea_create(panel);
    lv_obj_set_size(ta_name, SIDE_PANEL_W - 24, 44);
    lv_obj_align(ta_name, LV_ALIGN_TOP_MID, 0, 128);
    lv_textarea_set_one_line(ta_name, true);
    lv_textarea_set_max_length(ta_name, 31);
    lv_textarea_set_placeholder_text(ta_name, "点击输入");
    lv_obj_set_style_bg_color(ta_name, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_border_color(ta_name, lv_color_hex(0x377dff), LV_PART_MAIN);
    lv_obj_set_style_border_width(ta_name, 2, LV_PART_MAIN);
    lv_obj_set_style_radius(ta_name, 6, LV_PART_MAIN);
    lv_obj_add_event_cb(ta_name, name_textarea_cb, LV_EVENT_ALL, NULL);

    lv_obj_t *enroll = make_button(panel, SIDE_PANEL_W - 24, 48, "录入");
    lv_obj_align(enroll, LV_ALIGN_TOP_MID, 0, 184);
    lv_obj_add_event_cb(enroll, enroll_cb, LV_EVENT_CLICKED, NULL);

    lv_obj_t *live_label = lv_label_create(panel);
    set_label(live_label, "实时");
    lv_obj_align(live_label, LV_ALIGN_TOP_LEFT, 0, 248);

    sw_live = lv_switch_create(panel);
    lv_obj_align(sw_live, LV_ALIGN_TOP_RIGHT, 0, 238);
    lv_obj_add_event_cb(sw_live, sw_live_cb, LV_EVENT_VALUE_CHANGED, NULL);

    lbl_db_count = lv_label_create(panel);
    set_label(lbl_db_count, "特征库: 0");
    lv_obj_set_width(lbl_db_count, SIDE_PANEL_W - 24);
    lv_obj_align(lbl_db_count, LV_ALIGN_TOP_LEFT, 0, 302);

    ta_del_id = lv_textarea_create(panel);
    lv_obj_set_size(ta_del_id, SIDE_PANEL_W - 24, 44);
    lv_obj_align(ta_del_id, LV_ALIGN_TOP_MID, 0, 328);
    lv_textarea_set_one_line(ta_del_id, true);
    lv_textarea_set_accepted_chars(ta_del_id, "0123456789");
    lv_textarea_set_max_length(ta_del_id, 5);
    lv_textarea_set_placeholder_text(ta_del_id, "ID");
    lv_obj_set_style_bg_color(ta_del_id, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_border_color(ta_del_id, lv_color_hex(0x00aa33), LV_PART_MAIN);
    lv_obj_set_style_border_width(ta_del_id, 2, LV_PART_MAIN);
    lv_obj_set_style_radius(ta_del_id, 6, LV_PART_MAIN);
    lv_obj_add_event_cb(ta_del_id, name_textarea_cb, LV_EVENT_ALL, NULL);

    lv_obj_t *del_id = make_button(panel, SIDE_PANEL_W - 24, 44, "删除ID");
    lv_obj_align(del_id, LV_ALIGN_TOP_MID, 0, 384);
    lv_obj_add_event_cb(del_id, delete_id_cb, LV_EVENT_CLICKED, NULL);

    lv_obj_t *del_last = make_button(panel, SIDE_PANEL_W - 24, 44, "删除最后");
    lv_obj_align(del_last, LV_ALIGN_TOP_MID, 0, 436);
    lv_obj_add_event_cb(del_last, delete_last_cb, LV_EVENT_CLICKED, NULL);

    lv_obj_t *clear = make_button(panel, SIDE_PANEL_W - 24, 44, "清空");
    lv_obj_align(clear, LV_ALIGN_TOP_MID, 0, 488);
    lv_obj_add_event_cb(clear, clear_db_cb, LV_EVENT_CLICKED, NULL);

    lbl_status = lv_label_create(panel);
    lv_obj_set_width(lbl_status, SIDE_PANEL_W - 24);
    lv_label_set_long_mode(lbl_status, LV_LABEL_LONG_DOT);
    set_label(lbl_status, "准备就绪");
    lv_obj_set_style_bg_color(lbl_status, lv_color_white(), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(lbl_status, LV_OPA_80, LV_PART_MAIN);
    lv_obj_set_style_text_color(lbl_status, lv_color_hex(0x263244), LV_PART_MAIN);
    lv_obj_set_style_pad_all(lbl_status, 8, LV_PART_MAIN);
    lv_obj_set_style_radius(lbl_status, 8, LV_PART_MAIN);
    lv_obj_align(lbl_status, LV_ALIGN_BOTTOM_MID, 0, 0);

    kb_name = lv_keyboard_create(scr_face);
    lv_obj_set_size(kb_name, SCREEN_W, 230);
    lv_obj_align(kb_name, LV_ALIGN_BOTTOM_MID, 0, 0);
    lv_obj_add_flag(kb_name, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_event_cb(kb_name, name_textarea_cb, LV_EVENT_READY, NULL);
    lv_obj_add_event_cb(kb_name, name_textarea_cb, LV_EVENT_CANCEL, NULL);
}

static void boot_logo_opa_anim_cb(void *obj, int32_t value)
{
    if (obj) {
        lv_obj_set_style_opa((lv_obj_t *)obj, value, LV_PART_MAIN);
    }
}

static lv_obj_t *boot_label(lv_obj_t *parent,
                            const char *text,
                            int32_t x,
                            int32_t y,
                            int32_t w,
                            const lv_font_t *font,
                            uint32_t color)
{
    lv_obj_t *label = lv_label_create(parent);
    if (!label) {
        mark_lvgl_build_failed("boot label");
        return nullptr;
    }
    lv_obj_set_pos(label, x, y);
    lv_obj_set_width(label, w);
    lv_label_set_long_mode(label, LV_LABEL_LONG_DOT);
    lv_label_set_text(label, text ? text : "");
    lv_obj_set_style_text_font(label, font, LV_PART_MAIN);
    lv_obj_set_style_text_color(label, lv_color_hex(color), LV_PART_MAIN);
    lv_obj_set_style_text_letter_space(label, 0, LV_PART_MAIN);
    lv_obj_set_style_text_line_space(label, 5, LV_PART_MAIN);
    return label;
}

static void create_boot_screen(void)
{
    scr_boot = lv_obj_create(NULL);
    if (!scr_boot) {
        mark_lvgl_build_failed("boot screen");
        return;
    }
    lv_obj_clear_flag(scr_boot, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_set_size(scr_boot, SCREEN_W, SCREEN_H);
    lv_obj_set_style_bg_color(scr_boot, lv_color_hex(0x0d1822), LV_PART_MAIN);
    lv_obj_set_style_pad_all(scr_boot, 0, LV_PART_MAIN);

    lv_obj_t *accent = lv_obj_create(scr_boot);
    if (accent) {
        lv_obj_remove_style_all(accent);
        lv_obj_set_pos(accent, 0, 0);
        lv_obj_set_size(accent, 8, SCREEN_H);
        lv_obj_set_style_bg_color(accent, lv_color_hex(0x22a06b), LV_PART_MAIN);
        lv_obj_set_style_bg_opa(accent, LV_OPA_COVER, LV_PART_MAIN);
    }

    s_boot_logo = lv_image_create(scr_boot);
    if (s_boot_logo) {
        lv_image_set_src(s_boot_logo, &img_competition_logo);
        lv_obj_set_pos(s_boot_logo, 72, 82);
        lv_obj_set_style_opa(s_boot_logo, LV_OPA_TRANSP, LV_PART_MAIN);

        lv_anim_t anim;
        lv_anim_init(&anim);
        lv_anim_set_var(&anim, s_boot_logo);
        lv_anim_set_exec_cb(&anim, boot_logo_opa_anim_cb);
        lv_anim_set_values(&anim, LV_OPA_TRANSP, LV_OPA_COVER);
        lv_anim_set_time(&anim, 700);
        lv_anim_set_delay(&anim, 120);
        lv_anim_set_path_cb(&anim, lv_anim_path_ease_out);
        lv_anim_start(&anim);
    }

    boot_label(scr_boot, "智能健身镜", 72, 152, 360, font_18(), 0xf5fbff);
    boot_label(scr_boot, "姿态纠正训练系统", 72, 188, 360, font_14(), 0x9fb2bf);
    boot_label(scr_boot, "ESP32-P4 / LVGL / MoveNet / PC Pose", 72, 232, 380, font_14(), 0x6f8795);

    lv_obj_t *line = lv_obj_create(scr_boot);
    if (line) {
        lv_obj_remove_style_all(line);
        lv_obj_set_pos(line, 72, 284);
        lv_obj_set_size(line, 280, 2);
        lv_obj_set_style_bg_color(line, lv_color_hex(0x294655), LV_PART_MAIN);
        lv_obj_set_style_bg_opa(line, LV_OPA_COVER, LV_PART_MAIN);
    }
    boot_label(scr_boot, "系统启动中", 72, 312, 240, font_18(), 0xf5fbff);
    boot_label(scr_boot, "正在初始化本地界面、网络服务与姿态推理任务。", 72, 348, 390, font_14(), 0x9fb2bf);

    lv_obj_t *console = lv_obj_create(scr_boot);
    if (console) {
        lv_obj_set_pos(console, 516, 68);
        lv_obj_set_size(console, 440, 354);
        lv_obj_clear_flag(console, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_set_style_bg_color(console, lv_color_hex(0x101f2a), LV_PART_MAIN);
        lv_obj_set_style_border_color(console, lv_color_hex(0x294655), LV_PART_MAIN);
        lv_obj_set_style_border_width(console, 1, LV_PART_MAIN);
        lv_obj_set_style_radius(console, 8, LV_PART_MAIN);
        lv_obj_set_style_pad_all(console, 18, LV_PART_MAIN);

        boot_label(console, "BOOT CONSOLE", 0, 0, 190, font_14(), 0x22a06b);
        boot_label(console, "STATUS", 318, 0, 74, font_14(), 0x9fb2bf);
        for (int i = 0; i < 7; i++) {
            s_boot_log_labels[i] = boot_label(console, "", 0, 44 + i * 39, 390, font_14(), 0xd5e2e8);
        }
    }

    s_boot_bar = lv_bar_create(scr_boot);
    if (s_boot_bar) {
        lv_obj_set_pos(s_boot_bar, 72, 474);
        lv_obj_set_size(s_boot_bar, 884, 18);
        lv_bar_set_range(s_boot_bar, 0, 100);
        lv_bar_set_value(s_boot_bar, 0, LV_ANIM_OFF);
        lv_obj_set_style_radius(s_boot_bar, 9, LV_PART_MAIN);
        lv_obj_set_style_radius(s_boot_bar, 9, LV_PART_INDICATOR);
        lv_obj_set_style_bg_color(s_boot_bar, lv_color_hex(0x233541), LV_PART_MAIN);
        lv_obj_set_style_bg_color(s_boot_bar, lv_color_hex(0x22a06b), LV_PART_INDICATOR);
        lv_obj_set_style_border_width(s_boot_bar, 0, LV_PART_MAIN);
    }
    s_boot_percent = boot_label(scr_boot, "0%", 72, 508, 90, font_18(), 0xf5fbff);
    s_boot_stage = boot_label(scr_boot, "等待启动", 164, 510, 520, font_14(), 0x9fb2bf);
}

static void boot_update(int percent, const char *stage, const char *detail)
{
    if (!scr_boot) {
        return;
    }

    if (bsp_display_lock(pdMS_TO_TICKS(250))) {
        if (s_boot_bar) {
            lv_bar_set_value(s_boot_bar, percent, LV_ANIM_ON);
        }
        if (s_boot_percent) {
            char pct[16];
            snprintf(pct, sizeof(pct), "%d%%", percent);
            lv_label_set_text(s_boot_percent, pct);
        }
        if (s_boot_stage) {
            char stage_line[128];
            snprintf(stage_line, sizeof(stage_line), "%s", stage ? stage : "");
            lv_label_set_text(s_boot_stage, stage_line);
        }

        if (stage && detail) {
            if (s_boot_log_count < (int)(sizeof(s_boot_log_labels) / sizeof(s_boot_log_labels[0]))) {
                s_boot_log_count++;
            }
            for (int i = s_boot_log_count - 1; i > 0; i--) {
                if (s_boot_log_labels[i] && s_boot_log_labels[i - 1]) {
                    lv_label_set_text(s_boot_log_labels[i], lv_label_get_text(s_boot_log_labels[i - 1]));
                }
            }
            if (s_boot_log_labels[0]) {
                char line[160];
                snprintf(line, sizeof(line), "[%3d%%] %-12s %s", percent, stage, detail);
                lv_label_set_text(s_boot_log_labels[0], line);
            }
        }
        bsp_display_unlock();
    }

    vTaskDelay(pdMS_TO_TICKS(110));
}

extern "C" void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ESP_ERROR_CHECK(nvs_flash_init());
    }
    load_wifi_credentials();
    load_auth_account();
    load_user_profile();
    load_training_record();
    load_training_history();
    load_training_goals();
    const esp_reset_reason_t reset_reason = esp_reset_reason();
    ESP_LOGI(TAG, "Reset reason: %s (%d)", reset_reason_name(reset_reason), (int)reset_reason);
    (void)ensure_net_stack_ready();

    ESP_ERROR_CHECK(bsp_spiffs_mount());

    bsp_display_cfg_t cfg = {
        .lvgl_port_cfg = ESP_LVGL_PORT_INIT_CONFIG(),
        .buffer_size = BSP_LCD_H_RES * 40,
        .double_buffer = BSP_LCD_DRAW_BUFF_DOUBLE,
        .flags = {
            .buff_dma = false,
            .buff_spiram = true,
            .sw_rotate = true,
        },
    };
    cfg.lvgl_port_cfg.task_stack = 12288;

    s_disp = bsp_display_start_with_config(&cfg);
    if (s_disp && bsp_display_lock(portMAX_DELAY)) {
        bsp_display_rotate(s_disp, LV_DISPLAY_ROTATION_180);
        create_boot_screen();
        if (scr_boot) {
            lv_scr_load(scr_boot);
        }
        bsp_display_unlock();
    }
    bsp_display_backlight_on();
    boot_update(10, "Storage", "NVS ready");
    boot_update(20, "Filesystem", "SPIFFS mounted");
    esp_err_t sd_err = bsp_sdcard_mount();
    s_sdcard_mounted = (sd_err == ESP_OK);
    bool sdcard_retry_needed = !s_sdcard_mounted;
    if (s_sdcard_mounted) {
        video_reload_manifests();
        boot_update(26, "TF Card", "music library storage mounted");
    } else {
        ESP_LOGW(TAG, "TF/SD card mount failed: %s", esp_err_to_name(sd_err));
        boot_update(26, "TF Card", "mount pending, retry in background");
    }
    boot_update(32, "Display", "LVGL panel online");

    s_uiq = xQueueCreate(8, sizeof(ui_msg_t));
    s_model_mutex = xSemaphoreCreateMutex();
    s_overlay_mutex = xSemaphoreCreateMutex();
    s_pose_overlay_mutex = xSemaphoreCreateMutex();
    s_music_mutex = xSemaphoreCreateMutex();
    s_weather_mutex = xSemaphoreCreateMutex();
    s_training_mutex = xSemaphoreCreateMutex();

    s_db_count.store(0);
    boot_update(42, "Kernel", "queues and mutexes ready");
    music_start_service();
    start_weather_task();
    if (sdcard_retry_needed) {
        start_sdcard_retry_task();
    }

    if (bsp_display_lock(portMAX_DELAY)) {
        create_page_checked(&scr_menu, create_menu, "overview");
        create_page_checked(&scr_auth, create_auth_page, "auth");
        lv_timer_create(ui_timer_cb, 50, nullptr);
        bsp_display_unlock();
    }
    boot_update(54, "UI", "auth and overview pages built");

    bool preload_ok = false;
    if (bsp_display_lock(portMAX_DELAY)) {
        preload_ok = preload_all_product_pages_now();
        if (!preload_ok) {
            start_page_preloader();
        }
        bsp_display_unlock();
    }
    boot_update(preload_ok ? 70 : 66, "UI", preload_ok ? "pages preloaded" : "background preload enabled");

    if (!s_live_sem) {
        s_live_sem = xSemaphoreCreateBinary();
    }
    if (!s_enroll_sem) {
        s_enroll_sem = xSemaphoreCreateBinary();
    }
    if (!s_pose_sem) {
        s_pose_sem = xSemaphoreCreateBinary();
    }
    if (!s_body_roi_sem) {
        s_body_roi_sem = xSemaphoreCreateBinary();
    }
    if (!s_pc_pose_sem) {
        s_pc_pose_sem = xSemaphoreCreateBinary();
    }
    boot_update(80, "RTOS", "worker semaphores ready");

    boot_update(86, "Camera", "deferred until face or training");

    xTaskCreatePinnedToCore(face_worker_task, "face_worker", 20 * 1024, nullptr, 1, &s_worker_task, 1);
    xTaskCreatePinnedToCore(pc_pose_worker_task, "pc_pose_worker", 10 * 1024, nullptr, 2, &s_pc_pose_task, 1);
    boot_update(92, "Workers", "vision and pose workers online");

    xTaskCreatePinnedToCore(pc_pose_wifi_task, "pc_pose_wifi", 6144, nullptr, 1, nullptr, 0);
    boot_update(96, "Network", "PC pose Wi-Fi task started");

    xTaskCreatePinnedToCore(wifi_watchdog_task, "wifi_watchdog", 4096, nullptr, 1, &s_wifi_watchdog_task, 0);
    boot_update(97, "Network", "Wi-Fi watchdog started");

    if (RUN_STATIC_POSE_SELF_TEST) {
        run_static_pose_self_test();
    }
    boot_update(100, "Ready", "entering auth UI");
    if (bsp_display_lock(portMAX_DELAY)) {
        g_ui_screen.store(static_cast<int>(UiScreen::Auth));
        if (!scr_auth) {
            create_page_checked(&scr_auth, create_auth_page, "auth");
        }
        lv_obj_t *target = scr_auth ? scr_auth : scr_menu;
        if (target) {
            lv_scr_load_anim(target, LV_SCREEN_LOAD_ANIM_FADE_ON, 320, 0, false);
        }
        bsp_display_unlock();
    }

    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
