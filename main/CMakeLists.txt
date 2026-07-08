idf_component_register(
    SRCS "main.c" "main.cpp" "logo_lvgl.c"
         "mjpeg_player/esp_lvgl_simple_player.c"
         "mjpeg_player/media_src_storage.c"
         "fonts/font_zh_14.c" "fonts/font_zh_16.c" "fonts/font_zh_18.c"
    INCLUDE_DIRS "."
    REQUIRES human_face_detect human_pose_detect pedestrian_detect camera_port esp32_p4_wifi6_touch_lcd_7b esp_video
             espressif__human_face_recognition nvs_flash esp_timer driver esp_mm
             esp_wifi esp_event esp_netif esp_wifi_remote lwip
             esp_lvgl_port esp_lcd_touch_gt911 lvgl__lvgl esp-dl esp_driver_jpeg esp_driver_ppa
             espressif__esp_codec_dev fatfs
    EMBED_FILES "test_person.jpg"
)

target_compile_options(${COMPONENT_LIB} PRIVATE -Wno-missing-field-initializers)
