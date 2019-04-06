project(easyloggingpp)
cmake_minimum_required(VERSION 3.2)
find_package(Git)

## STAMPA TUTTE LE VARIABILI DI AMBIENTE CREATE DA CMAKE
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

include(ExternalProject)

ExternalProject_Add(easyloggingpp
    GIT_REPOSITORY https://github.com/muflihun/easyloggingpp.git
	#PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
	BUILD_IN_SOURCE 1
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CONFIGURE_COMMAND ""
	#INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

ExternalProject_Add_Step(
    easyloggingpp copySource
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/easyloggingpp-prefix/src/easyloggingpp/src/easylogging++.cc ${CMAKE_CURRENT_SOURCE_DIR}/easyloggingpp/.
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/easyloggingpp-prefix/src/easyloggingpp/src/easylogging++.h ${CMAKE_CURRENT_SOURCE_DIR}/easyloggingpp/.
    DEPENDEES install
)

#set(glfw_LIB ${CMAKE_CURRENT_BINARY_DIR}/glfw-prefix/src/glew/src/libglfw3.a)
