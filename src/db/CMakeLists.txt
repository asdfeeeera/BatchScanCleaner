add_library(db STATIC)
target_include_directories(db PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(db PUBLIC Qt5::Core Qt5::Sql)