add_library(ui STATIC
    main_window.h
    main_window.cpp
    settings_dialog.h
    settings_dialog.cpp
)

target_include_directories(ui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/image
    ${CMAKE_SOURCE_DIR}/src/core
    ${CMAKE_SOURCE_DIR}/src/process
    ${CMAKE_SOURCE_DIR}/src/protect
    ${CMAKE_SOURCE_DIR}/src/analyze
    ${CMAKE_SOURCE_DIR}/src/db
)

target_link_libraries(ui PUBLIC
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    image
    core
    process
    protect
    analyze
    db
)