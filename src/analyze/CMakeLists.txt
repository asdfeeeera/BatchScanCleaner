add_library(analyze STATIC)
target_include_directories(analyze PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(analyze PUBLIC core image Qt5::Core)