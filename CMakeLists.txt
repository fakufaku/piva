# https://stackoverflow.com/questions/51907755/building-a-pybind11-module-with-cpp-and-cuda-sources-using-cmake

cmake_minimum_required(VERSION 3.1)

set(CMAKE_VERBOSE_MAKEFILE ON)

# We want to use the Python (anaconda) prefix
if (WIN32)
  get_filename_component(PYTHON_PREFIX ${PYTHON_EXECUTABLE} DIRECTORY)
else()
  exec_program("which python | sed 's:/bin/python::'"
    OUTPUT_VARIABLE PYTHON_PREFIX
    RETURN_VALUE PYTHON_NOT_FOUND
    )
  if(PYTHON_NOT_FOUND)
      message(FATAL_ERROR "Python prefix not found")
  endif()
endif()

message(STATUS "Found Python prefix ${PYTHON_PREFIX}")
set(CMAKE_PREFIX_PATH ${PYTHON_PREFIX} ${CMAKE_PREFIX_PATH})

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
message("The cmake module path ${CMAKE_MODULE_PATH}")
message("The current source dir ${CMAKE_CURRENT_SOURCE_DIR}")

project(core)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PythonInterp)

# This goes after, since it uses PythonInterp as hint
find_package(PythonLibs)
find_package(Numpy)
find_package(FFTW)
#find_package(MKL)

# Use just one of these:
if (MSVC)
  set(PYBIND11_CPP_STANDARD /std:c++14)
else()
  # GCC/clang:
  set(PYBIND11_CPP_STANDARD -std=c++14)
endif()
find_package(pybind11)

pybind11_add_module(core
  src/main.cpp
  )
target_include_directories(core PRIVATE ${PYTHON_INCLUDE_DIRS})
target_include_directories(core PRIVATE ${NUMPY_INCLUDE_DIRS})
target_include_directories(core PRIVATE ./include)

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

# We use TBB, but turn it off in XTENSOR
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
find_package(TBB REQUIRED)
message(STATUS "Found intel TBB: ${TBB_INCLUDE_DIRS}")
message(STATUS "TBB Libraries: ${TBB_LIBRARIES}")

if(MSVC)
    target_compile_options(core PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(core PRIVATE -std=c++14 -O3 -Wall -Wextra)
    # target_compile_options(core PRIVATE -std=c++14 -Wall -Wextra -g)
endif()

add_definitions(-DHAVE_CBLAS=1)
# add_definitions(-DXTENSOR_USE_TBB)
add_definitions(-DXTENSOR_USE_SIMD)
add_definitions(-DVERBOSE=1)

if (WIN32)
  # On windows, MKL does not seem always available for some reason
  # use OpenBLAS instead
  set(BLA_VENDOR OpenBLAS)
else()
  # To use MKL Blas
  # If possible, we want the sequential MKL version
  # so that it doesn't interfer with parallelization with TBB
  set(BLA_VENDOR Intel10_64lp_seq)
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
set(BLAS_LIBRARIES ${BLAS_LIBRARIES})

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

### Final target setup
set_target_properties(
    core
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "core"
        LINKER_LANGUAGE C
    )

target_link_libraries(core PUBLIC xtensor ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${FFTW_LIBRARIES} ${TBB_LIBRARIES})
