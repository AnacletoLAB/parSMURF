project(jsoncons)
cmake_minimum_required(VERSION 3.2)
find_package(Git)

### STAMPA TUTTE LE VARIABILI DI AMBIENTE CREATE DA CMAKE
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

include(ExternalProject)

ExternalProject_Add(jsoncons
    GIT_REPOSITORY https://github.com/danielaparker/jsoncons
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

ExternalProject_Add_Step(
    jsoncons copyFiles
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/jsoncons-prefix/src/jsoncons/include/jsoncons ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDEES install
)
