add_library(protect STATIC)
target_include_directories(protect PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(protect PUBLIC core image Qt5::Core)