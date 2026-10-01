add_library(image STATIC
    image_io.h
    image_io.cpp
    jpeg_writer.h
    jpeg_writer.cpp
)

target_include_directories(image PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(image PUBLIC
    Qt5::Core
    Qt5::Gui
    ${OpenCV_LIBS}
)