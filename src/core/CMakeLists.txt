add_library(core STATIC
    file_scanner.h
    file_scanner.cpp
)

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(core PUBLIC Qt5::Core)