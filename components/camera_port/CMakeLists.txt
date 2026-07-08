idf_component_register(
    SRCS "app_video.c" "app_camera_pipeline.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_video driver freertos esp_timer esp_mm esp-dl
)
