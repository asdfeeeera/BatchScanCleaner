add_library(backup STATIC)
target_include_directories(backup PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(backup PUBLIC core image Qt5::Core)