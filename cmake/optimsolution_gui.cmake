# optimsolution_gui.cmake
# Injects a Qt Widgets GUI target without modifying existing project files.
#
# Usage:
#   cmake -S . -B build "-DCMAKE_PROJECT_INCLUDE:FILEPATH=$PWD/cmake/optimsolution_gui.cmake" -DCMAKE_PREFIX_PATH="C:\Qt\6.x.x\msvc2022_64"
#
# Then:
#   cmake --build build --config Release --target optimsolution_gui

if (TARGET optimsolution_gui)
  return()
endif()

find_package(Qt6 COMPONENTS Widgets REQUIRED)

set(GUI_DIR "${CMAKE_SOURCE_DIR}/gui")

add_executable(optimsolution_gui
  "${GUI_DIR}/optimsolution_gui_main.cpp"
  "${GUI_DIR}/MainWindow.cpp"
  "${GUI_DIR}/MainWindow.h"
  "${GUI_DIR}/PathUtils.cpp"
  "${GUI_DIR}/PathUtils.h"
  "${GUI_DIR}/FactoryIntrospect.cpp"
  "${GUI_DIR}/FactoryIntrospect.h"
  "${GUI_DIR}/ConfigFile.cpp"
  "${GUI_DIR}/ConfigFile.h"
  "${GUI_DIR}/AnsiStrip.cpp"
  "${GUI_DIR}/AnsiStrip.h"
  "${GUI_DIR}/CrashLog.cpp"
  "${GUI_DIR}/CrashLog.h"
)

target_compile_features(optimsolution_gui PRIVATE cxx_std_17)
set_target_properties(optimsolution_gui PROPERTIES
  AUTOMOC ON
  AUTOUIC ON
  AUTORCC OFF
)

target_include_directories(optimsolution_gui PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${GUI_DIR}"
)

target_link_libraries(optimsolution_gui PRIVATE
  Qt6::Widgets
  optimcore
)

if (MSVC)
  target_compile_options(optimsolution_gui PRIVATE /Zc:__cplusplus)
endif()

# Ensure the CLI executable is available next to the GUI.
if (TARGET optimsolution)
  add_custom_command(TARGET optimsolution_gui POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:optimsolution>
            $<TARGET_FILE_DIR:optimsolution_gui>/$<TARGET_FILE_NAME:optimsolution>
    COMMENT "Copying optimsolution next to optimsolution_gui"
  )
endif()
