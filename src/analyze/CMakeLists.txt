add_library(analyze STATIC
    pending_item.h
    pending_center.h
    pending_center.cpp
    pending_dialog.h
    pending_dialog.cpp
)
target_include_directories(analyze PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(analyze PUBLIC core image Qt5::Core Qt5::Gui Qt5::Widgets)