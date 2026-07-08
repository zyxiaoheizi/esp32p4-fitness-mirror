# ESP-IDF 工程：配置前请设置环境变量 IDF_PATH（指向 esp-idf 根目录）。
# 使用裸 CMake 时还需安装 Ninja，并让 CMAKE_MAKE_PROGRAM 指向 ninja.exe；
# 推荐在 VS Code 中用「ESP-IDF: 构建」任务，扩展会自动传入与官方 idf.py 相同的参数。
cmake_minimum_required(VERSION 3.16)

set(CMAKE_OBJECT_PATH_MAX 512)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

add_compile_options(
    -Wno-format
    -Wno-unused-variable
    -Wno-missing-field-initializers
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-enum-enum-conversion -Wno-write-strings")

project(final_projct)
