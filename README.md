# ESP32-P4 智能健身镜关键代码

本目录整理了项目中与固件编译、界面显示、姿态检测、离线处理和训练记录相关的主要代码。目录内容按原工程结构保留，便于查看和迁移。

## 目录结构

- `main/`：板端主程序、LVGL 界面、字体资源、MJPEG 播放器和嵌入资源。
- `components/`：摄像头、姿态检测、人体检测和人脸相关组件源码。
- `tools/pc_pose_server/`：PC 端姿态服务核心程序。
- `CMakeLists.txt`、`sdkconfig`、`partitions.csv`：ESP-IDF 工程配置文件。

## 使用方式

1. 准备 ESP-IDF v5.4.x 环境，并确认目标芯片为 `esp32p4`。
2. 将本目录内容放入 ESP-IDF 工程根目录，保持 `main/`、`components/` 和配置文件的相对路径不变。
3. 根据工程需要补齐模型文件、SD 卡资源和组件管理器下载的依赖组件。
4. 在工程根目录执行：

```powershell
idf.py set-target esp32p4
idf.py build
idf.py flash monitor
```

## 说明

本目录主要用于代码查阅、报告提交和核心逻辑备份。模型权重、课程视频、音乐、训练视频、构建产物和启动脚本未包含在内。
