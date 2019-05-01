# https://stackoverflow.com/questions/51907755/building-a-pybind11-module-with-cpp-and-cuda-sources-using-cmake
cmake_minimum_required(VERSION 3.1)
project(piva-examples-cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# We want to use the Python (anaconda) prefix
exec_program("which python | sed 's:/bin/python::'"
  OUTPUT_VARIABLE PYTHON_PREFIX
  RETURN_VALUE PYTHON_NOT_FOUND
  )
if(PYTHON_NOT_FOUND)
    message(FATAL_ERROR "Python prefix not found")
endif()
message(STATUS "Found Python prefix ${PYTHON_PREFIX}")
set(CMAKE_PREFIX_PATH ${PYTHON_PREFIX} ${CMAKE_PREFIX_PATH})

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
message("The cmake module path ${CMAKE_MODULE_PATH}")
message("The current source dir ${CMAKE_CURRENT_SOURCE_DIR}")
message("CMake prefix path ${CMAKE_PREFIX_PATH}")

find_package(FFTW)
message(STATUS "Found FFTW ${FFTW_LIBRARIES}")
find_library(SNDFILE_LIBRARIES sndfile)
message(STATUS "Found sndfile ${SNDFILE_LIBRARIES}")

#### XTENSOR STUFF
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

OPTION(XTENSOR_USE_XSIMD "simd acceleration for xtensor" ON)
OPTION(XTENSOR_USE_TBB "enable parallelization using intel TBB" OFF)

if(XTENSOR_USE_XSIMD)
    set(xsimd_REQUIRED_VERSION 7.0.0)
    find_package(xsimd ${xsimd_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xsimd: ${xsimd_INCLUDE_DIRS}/xsimd")
endif()

#### TBB
find_package(TBB REQUIRED)
message(STATUS "Found intel TBB: ${TBB_INCLUDE_DIRS}")

#### BLAS
add_definitions(-DHAVE_CBLAS=1)
# add_definitions(-DXTENSOR_USE_TBB)
add_definitions(-DXTENSOR_USE_SIMD)
add_definitions(-DVERBOSE=1)

if (WIN32)
    set(BLA_VENDOR OpenBLAS)
else()
    # If possible, we want the sequential MKL version
    # so that it doesn't interfer with parallelization with TBB
    set(BLA_VENDOR Intel10_64lp_seq)
endif()
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

#### Configure all targets
set(APP_SOURCES src/auxiva.cpp src/auxiva_iss.cpp)

foreach( sourcefile ${APP_SOURCES} )
    # I used a simple string replace, to cut off .cpp.
    string( REPLACE ".cpp" "" targetname ${sourcefile} )
    string( REPLACE "src/" "" targetname ${targetname} )
    add_executable( ${targetname} ${sourcefile} )

    target_include_directories(${targetname} PRIVATE ./include)
    target_include_directories(${targetname} PUBLIC "../include")

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
        CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
        (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
        target_compile_options(${targetname} PRIVATE -O3 -march=native -std=c++14 -Wall -Wextra)
    endif()

    # Make sure libs are linked to each app
    target_link_libraries(${targetname} xtensor ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${FFTW_LIBRARIES} ${SNDFILE_LIBRARIES} ${TBB_LIBRARIES})
endforeach( sourcefile ${APP_SOURCES} )
