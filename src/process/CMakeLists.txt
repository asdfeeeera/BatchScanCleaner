add_library(process STATIC)
target_include_directories(process PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(process PUBLIC core image analyze Qt5::Core)