add_library(config STATIC)
target_include_directories(config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(config PUBLIC Qt5::Core)