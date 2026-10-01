add_library(image STATIC)
target_include_directories(image PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(image PUBLIC Qt5::Core)