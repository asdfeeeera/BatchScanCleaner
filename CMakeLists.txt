cmake_minimum_required(VERSION 3.20)

project(BatchScanCleaner
    VERSION 1.0.0.0
    DESCRIPTION "批量扫描图片净化增强软件"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(MSVC)
    add_compile_options(/W4 /utf-8 /MP)
    add_compile_definitions(
        UNICODE
        _UNICODE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
        _WIN32_WINNT=0x0601
        WINVER=0x0601
    )
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# 本地 Qt 路径
if(EXISTS "E:/Qt/5.14.2/msvc2017_64/lib/cmake/Qt5")
    set(Qt5_DIR "E:/Qt/5.14.2/msvc2017_64/lib/cmake/Qt5" CACHE PATH "Qt5 CMake 配置目录")
endif()

find_package(Qt5 5.14 REQUIRED COMPONENTS
    Core Gui Widgets Sql Concurrent
)

message(STATUS "Qt5 版本: ${Qt5_VERSION}")

# OpenCV
if(DEFINED ENV{OpenCV_DIR})
    set(OpenCV_DIR $ENV{OpenCV_DIR})
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs photo)
message(STATUS "OpenCV 版本: ${OpenCV_VERSION}")
message(STATUS "OpenCV 库: ${OpenCV_LIBS}")

add_subdirectory(src)

if(WIN32)
    get_target_property(QT_QMAKE_EXECUTABLE Qt5::qmake IMPORTED_LOCATION)
    get_filename_component(QT_BIN_DIR "${QT_QMAKE_EXECUTABLE}" DIRECTORY)

    add_custom_target(deploy_qt_plugins ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:BatchScanCleaner>/platforms"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${QT_BIN_DIR}/../plugins/platforms/qwindows.dll"
            "$<TARGET_FILE_DIR:BatchScanCleaner>/platforms/"
        COMMENT "部署 Qt 平台插件"
    )
endif()