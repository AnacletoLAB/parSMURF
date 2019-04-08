# parSMURF
# Alessandro Petrini, 2018-2019

project(parSMURF)
cmake_minimum_required(VERSION 2.8)

find_package(MPI)
if (MPI_FOUND)
	include_directories(${MPI_INCLUDE_PATH})
endif()

add_subdirectory(ann_1.1.2)
add_subdirectory(jsoncons)
add_subdirectory(zlib)
add_subdirectory(easyloggingpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
include_directories(ann_1.1.2/include/ANN)
include_directories(ranger)
include_directories(.)
include_directories(ranger/Utility)
include_directories(ranger/Forest)
include_directories(ranger/Tree)
include_directories(${jsoncons_SOURCE_DIR})
include_directories(${zlib_SOURCE_DIR})
include_directories(easyloggingpp)

SET(SRCCOMMON				"ArgHandler_new.cpp"
							"fileImport.cpp"
							"curves.cpp"
							"folds.cpp"
							"testtraindivider.cpp"
							"HyperSMURFUtils.cpp"
							"partition.cpp"
							"sampler.cpp"
							"rfRanger.cpp")

FILE(GLOB_RECURSE RANGER	"ranger/*.cpp")

SET(LOGGERSRC 				"easyloggingpp/easyloggingpp/easylogging++.cc")

SET(SOURCESGRID				"parSMURF.cpp"
							"samplerKNN.cpp"
							"hyperSMURF.cpp")

SET(SOURCESGRDMPI			"parSMURF_MPI.cpp"
							"hyperSMURF_MPI.cpp"
							"MPIHelper.cpp"
							"partitionMPI.cpp"
							"samplerKNNMPI.cpp"
							"sendRecvThrd.cpp")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

ADD_CUSTOM_TARGET(debug
	COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
	COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)
ADD_CUSTOM_TARGET(release
	COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
	COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)

add_executable(parSMURF1 ${SRCCOMMON} ${RANGER} ${SOURCESGRID})
add_dependencies(parSMURF1 ANN)
add_dependencies(parSMURF1 jsoncons)
add_dependencies(parSMURF1 zlib)
add_dependencies(parSMURF1 easyloggingpp)

if (MPI_FOUND)

	add_executable(parSMURFn ${SRCCOMMON} ${RANGER} ${SOURCESGRDMPI} ${LOGGERSRC})
	add_dependencies(parSMURFn ANN)
	add_dependencies(parSMURFn jsoncons)
	add_dependencies(parSMURFn zlib)
	add_dependencies(parSMURFn easyloggingpp)

	if(MPI_COMPILE_FLAGS)
		set_target_properties(parSMURFn PROPERTIES
			COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
	endif()

	if(MPI_LINK_FLAGS)
		set_target_properties(parSMURFn PROPERTIES
			LINK_FLAGS "${MPI_LINK_FLAGS}")
	endif()
endif()

target_link_libraries(parSMURF1 ANN ${zlib_LIB})
if (MPI_FOUND)
	target_link_libraries(parSMURFn ANN ${zlib_LIB} ${MPI_LIBRARIES})
endif()
