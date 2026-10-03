add_library(db STATIC
    batch_step.h
    batch_processor.h
    batch_processor.cpp
    batch_dialog.h
    batch_dialog.cpp
)

target_include_directories(db PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/image
    ${CMAKE_SOURCE_DIR}/src/process
    ${CMAKE_SOURCE_DIR}/src/protect
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(db PUBLIC
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    Qt5::Sql
    core
    image
    process
    protect
    ${OpenCV_LIBS}
)