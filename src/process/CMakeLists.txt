add_library(process STATIC
    deskew.h
    deskew.cpp
    blackedge.h
    blackedge.cpp
)

target_include_directories(process PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/image
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(process PUBLIC
    Qt5::Core
    core
    image
    ${OpenCV_LIBS}
)