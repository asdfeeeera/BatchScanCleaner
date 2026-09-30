add_executable(BatchScanCleaner
    main.cpp
)

target_link_libraries(BatchScanCleaner PRIVATE
    ui
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
)

if(MSVC)
    set_target_properties(BatchScanCleaner PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()