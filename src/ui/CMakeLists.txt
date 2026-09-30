add_library(ui STATIC
    main_window.h
    main_window.cpp
)

target_link_libraries(ui PUBLIC
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
)