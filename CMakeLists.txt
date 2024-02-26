cmake_minimum_required(VERSION 3.1)
project(yolov4_tiny)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Путь к установленному espeak
#set(espeak_DIR "/usr/include/")
#find_package(espeak REQUIRED)
#include_directories(${ESPEAK_INCLUDE_DIRS})

# Добавьте эту строку с правильным путем до заголовочных файлов Darknet
include_directories(/home/sie/darknet/include)

add_executable(yolov4_tiny main.cpp)

target_link_libraries(yolov4_tiny "/usr/lib/x86_64-linux-gnu/libespeak.so")
target_link_libraries(yolov4_tiny ${OpenCV_LIBS})
target_link_libraries(yolov4_tiny "/home/sie/darknet/libdarknet.so")

