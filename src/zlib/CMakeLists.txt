project(zlib)
cmake_minimum_required(VERSION 2.8)
find_package(Git)

## STAMPA TUTTE LE VARIABILI DI AMBIENTE CREATE DA CMAKE
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

include(ExternalProject)

ExternalProject_Add(zlib
    GIT_REPOSITORY https://github.com/madler/zlib
	#PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
	BUILD_IN_SOURCE 1
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CONFIGURE_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/zlib-prefix/src/zlib/configure"
	#INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}
    #BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

ExternalProject_Add_Step(
    zlib copyInclude
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/zlib-prefix/src/zlib/zlib.h ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDEES install
)

set(zlib_LIB ${CMAKE_CURRENT_BINARY_DIR}/zlib-prefix/src/zlib/libz.a)
