add_library(log STATIC)
target_include_directories(log PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(log PUBLIC Qt5::Core)