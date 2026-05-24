# cmake/optimsolution_gui.cmake
#
# Loaded via:
#   -DCMAKE_PROJECT_INCLUDE=cmake/optimsolution_gui.cmake
#
# Defines optimsolution_add_gui(), called by the top-level CMakeLists.txt after optimcore exists.

if(DEFINED OPTIMSOLUTION_GUI_CMAKE_INCLUDED)
    return()
endif()
set(OPTIMSOLUTION_GUI_CMAKE_INCLUDED 1)

function(optimsolution_add_gui)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)

    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTORCC ON)

    # REQUIRED GUI sources (must include the GUI entrypoint that defines main()).
    set(OPTIMSOLUTION_GUI_CPP
        "${CMAKE_SOURCE_DIR}/gui/optimsolution_gui_main.cpp"
        "${CMAKE_SOURCE_DIR}/gui/MainWindow.cpp"
        "${CMAKE_SOURCE_DIR}/gui/BusySpinner.cpp"
        "${CMAKE_SOURCE_DIR}/gui/AnsiStrip.cpp"
        "${CMAKE_SOURCE_DIR}/gui/ConfigFile.cpp"
        "${CMAKE_SOURCE_DIR}/gui/CrashLog.cpp"
        "${CMAKE_SOURCE_DIR}/gui/FactoryIntrospect.cpp"
        "${CMAKE_SOURCE_DIR}/gui/PathUtils.cpp"
        "${CMAKE_SOURCE_DIR}/gui/CodeGenDialog.cpp"
    )

    set(OPTIMSOLUTION_GUI_HDR
        "${CMAKE_SOURCE_DIR}/gui/MainWindow.h"
        "${CMAKE_SOURCE_DIR}/gui/BusySpinner.h"
        "${CMAKE_SOURCE_DIR}/gui/AnsiStrip.h"
        "${CMAKE_SOURCE_DIR}/gui/ConfigFile.h"
        "${CMAKE_SOURCE_DIR}/gui/CrashLog.h"
        "${CMAKE_SOURCE_DIR}/gui/FactoryIntrospect.h"
        "${CMAKE_SOURCE_DIR}/gui/PathUtils.h"
        "${CMAKE_SOURCE_DIR}/gui/CodeGenDialog.h"
    )

    # Resources: required for window/taskbar icon.
    set(OPTIMSOLUTION_GUI_RES
        "${CMAKE_SOURCE_DIR}/gui/resources.qrc"
    )

    # Windows executable icon (Explorer file icon).
    set(OPTIMSOLUTION_GUI_RC "")
    if(WIN32)
        set(OPTIMSOLUTION_GUI_RC "${CMAKE_SOURCE_DIR}/gui/optimsolution_gui.rc")
    endif()

    foreach(_f IN LISTS OPTIMSOLUTION_GUI_CPP OPTIMSOLUTION_GUI_HDR OPTIMSOLUTION_GUI_RES)
        if(NOT EXISTS "${_f}")
            message(FATAL_ERROR "Required GUI file not found: ${_f}")
        endif()
    endforeach()
    if(WIN32 AND NOT EXISTS "${OPTIMSOLUTION_GUI_RC}")
        message(FATAL_ERROR "Required Windows resource file not found: ${OPTIMSOLUTION_GUI_RC}")
    endif()

    add_executable(optimsolution_gui WIN32
        ${OPTIMSOLUTION_GUI_CPP}
        ${OPTIMSOLUTION_GUI_HDR}
        ${OPTIMSOLUTION_GUI_RES}
        ${OPTIMSOLUTION_GUI_RC}
    )
    set_target_properties(optimsolution_gui PROPERTIES WIN32_EXECUTABLE TRUE)

    target_link_libraries(optimsolution_gui PRIVATE optimcore Qt6::Widgets)

    target_include_directories(optimsolution_gui PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/gui"
    )

    if(MSVC)
        target_link_options(optimsolution_gui PRIVATE "/SUBSYSTEM:WINDOWS")
    endif()

    message(STATUS "optimsolution_gui: resources.qrc enabled; window icon will be loaded from :/icons/optimsolution.png")
endfunction()
