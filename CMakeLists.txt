cmake_minimum_required(VERSION 3.20)

project(BatchScanCleaner
    VERSION 1.0.0.0
    DESCRIPTION "批量扫描图片净化增强软件"
    LANGUAGES CXX
)

# C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# MSVC 设置
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

# Qt 自动处理
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Qt 5.14.2 路径
# 本地路径存在则使用；云端交给 find_package 自动查找
if(EXISTS "E:/Qt/5.14.2/msvc2017_64/lib/cmake/Qt5")
# 本地路径存在则使用；云端交给 find_package 自动查找
if(EXISTS "E:/Qt/5.14.2/msvc2017_64/lib/cmake/Qt5")
    set(Qt5_DIR "E:/Qt/5.14.2/msvc2017_64/lib/cmake/Qt5" CACHE PATH "Qt5 CMake 配置目录")
endif()
endif()

# 查找 Qt5
find_package(Qt5 5.14 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    Sql
    Concurrent
)

message(STATUS "Qt5 版本: ${Qt5_VERSION}")
message(STATUS "Qt5 路径: ${Qt5_DIR}")

# 子目录
add_subdirectory(src)

# 平台插件部署（Windows）
if(WIN32)
    get_target_property(QT_QMAKE_EXECUTABLE Qt5::qmake IMPORTED_LOCATION)
    get_filename_component(QT_BIN_DIR "${QT_QMAKE_EXECUTABLE}" DIRECTORY)

    add_custom_target(deploy_qt_plugins ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:BatchScanCleaner>/platforms"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${QT_BIN_DIR}/../plugins/platforms/qwindows.dll"
            "$<TARGET_FILE_DIR:BatchScanCleaner>/platforms/"
        COMMENT "部署 Qt 平台插件 qwindows.dll"
    )
endif()