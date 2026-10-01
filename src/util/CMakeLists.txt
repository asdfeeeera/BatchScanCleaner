add_library(util STATIC)
target_include_directories(util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(util PUBLIC Qt5::Core)