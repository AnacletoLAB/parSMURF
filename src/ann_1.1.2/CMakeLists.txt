project(libANN)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(include)
include_directories(src)

file(GLOB_RECURSE SOURCELIBANN	"ANN.cpp" "brute.cpp" "kd_tree.cpp" "kd_util.cpp" "kd_split.cpp"
								"kd_dump.cpp" "kd_search.cpp" "kd_pr_search.cpp" "kd_fix_rad_search.cpp"
								"bd_tree.cpp" "bd_search.cpp" "bd_pr_search.cpp" "bd_fix_rad_search.cpp"
								"perf.cpp")

add_library(ANN ${SOURCELIBANN})
