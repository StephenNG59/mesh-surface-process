cmake_minimum_required(VERSION 3.10.0)

project(mesh-surface-process)

### Set C++ standard (global)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

### Configure output locations
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Print the build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
endif()
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

### Configure the compiler
# This is a basic, decent setup that should do something sane on most compilers

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

  # using Clang (linux or apple) or GCC
  message("Using clang/gcc compiler flags")
  SET(BASE_CXX_FLAGS "-std=c++11 -Wall -Wextra")
  SET(DISABLED_WARNINGS " -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-deprecated-declarations -Wno-missing-braces -Wno-unused-private-field")
  SET(TRACE_INCLUDES " -H -Wno-error=unused-command-line-argument")

  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message("Setting clang-specific options")
    SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -ferror-limit=3 -fcolor-diagnostics")
    SET(CMAKE_CXX_FLAGS_DEBUG          "-g3 -fsanitize=address -fno-limit-debug-info")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message("Setting gcc-specific options")
    SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fmax-errors=5")
    SET(CMAKE_CXX_FLAGS_DEBUG          "-g3")
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-maybe-uninitialized -Wno-format-zero-length -Wno-unused-but-set-parameter -Wno-unused-but-set-variable")
  endif()
  
  SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
  SET(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
  message("Using Visual Studio compiler flags")
  set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /W4")
  set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /MP") # parallel build
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4267\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4244\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4305\"")  # ignore truncation on initialization
  SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

  add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
  add_definitions(-DNOMINMAX)
  add_definitions(-D_USE_MATH_DEFINES)
else()
  # unrecognized
  message( FATAL_ERROR "Unrecognized compiler [${CMAKE_CXX_COMPILER_ID}]" )
endif()


# === Deps
add_subdirectory(deps/vcglib)
add_subdirectory(deps/geometry-central)
add_subdirectory(deps/polyscope)
# Libigl
set(DFETCHCONTENT_SOURCE_DIR_LIBIGL ${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl)
add_subdirectory(deps/libigl)


### === Build our project stuff

# 00_Playground
set(SRCS_00
    deps/vcglib/wrap/ply/plylib.cpp
    src/00_Playground.cpp
    src/mesh_processor.cpp
)
add_executable(00_Playground "${SRCS_00}")
target_include_directories(00_Playground PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(00_Playground PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/vcglib")
target_include_directories(00_Playground PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl/include")
target_include_directories(00_Playground PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/spectra/include")
target_link_libraries(00_Playground geometry-central polyscope igl::core)


# 01_Gradient
set(SRCS_01
    src/01_Gradient.cpp
)
add_executable(01_Gradient "${SRCS_01}")
target_link_libraries(01_Gradient geometry-central polyscope)


# 02_Mapping
set(SRCS_02
    deps/vcglib/wrap/ply/plylib.cpp
    src/02_Mapping.cpp
)
add_executable(02_Mapping "${SRCS_02}")
target_include_directories(02_Mapping PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/vcglib")
target_link_libraries(02_Mapping geometry-central polyscope)


# 03_DataPrepare
set(SRCS_03
    deps/vcglib/wrap/ply/plylib.cpp
    src/03_DataPrepare.cpp
)
add_executable(03_DataPrepare "${SRCS_03}")
target_include_directories(03_DataPrepare PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(03_DataPrepare PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/vcglib")
target_link_libraries(03_DataPrepare geometry-central polyscope)


# 04_MICCAI (visualize)
set(SRCS_04
    deps/vcglib/wrap/ply/plylib.cpp
    src/04_MICCAI.cpp
)
add_executable(04_MICCAI "${SRCS_04}")
target_include_directories(04_MICCAI PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(04_MICCAI PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/vcglib")
target_link_libraries(04_MICCAI geometry-central polyscope)

# 05_MICCAIPrepare
set(SRCS_05
    deps/vcglib/wrap/ply/plylib.cpp
    src/05_MICCAIPrepare.cpp
)
add_executable(05_MICCAIPrepare "${SRCS_05}")
target_include_directories(05_MICCAIPrepare PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(05_MICCAIPrepare PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/vcglib")
target_link_libraries(05_MICCAIPrepare geometry-central polyscope)

# A1_MeshCleaner
set(SRCS_A1
    #deps/vcglib/wrap/ply/plylib.cpp
    src/A1_MeshCleaner.cpp
)
add_executable(A1_MeshCleaner "${SRCS_A1}")
target_include_directories(A1_MeshCleaner PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(A1_MeshCleaner PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/vcglib")
target_include_directories(A1_MeshCleaner PUBLIC "${GC_EIGEN_LOCATION}")
#target_link_libraries(A1_MeshCleaner )

# A3_MeshMapper
set(SRCS_A3
    src/A3_MeshMapper.cpp
)
add_executable(A3_MeshMapper "${SRCS_A3}")
target_include_directories(A3_MeshMapper PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(A3_MeshMapper PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl/include")
target_include_directories(A3_MeshMapper PUBLIC "${GC_EIGEN_LOCATION}")
target_include_directories(A3_MeshMapper PUBLIC "${MY_CNPY_DIR}/include") #cnpy.h zlib.h zconf.h
target_include_directories(A3_MeshMapper PUBLIC "${MY_ZLIB_DIR}/include")
target_link_directories(A3_MeshMapper PUBLIC "${MY_CNPY_DIR}/lib") #cnpy.lib cnpy.dll
target_link_directories(A3_MeshMapper PUBLIC "${MY_ZLIB_DIR}/lib")
target_link_libraries(A3_MeshMapper igl::core cnpy zlib)

# A4_ImageExtractor
set(SRCS_A4
    src/A4_ImageExtractor.cpp
)
add_executable(A4_ImageExtractor "${SRCS_A4}")
target_include_directories(A4_ImageExtractor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(A4_ImageExtractor PUBLIC "${GC_EIGEN_LOCATION}")
target_include_directories(A4_ImageExtractor PUBLIC "${MY_CNPY_DIR}/include")
target_include_directories(A4_ImageExtractor PUBLIC "${MY_ZLIB_DIR}/include")
target_link_directories(A4_ImageExtractor PUBLIC "${MY_CNPY_DIR}/lib")
target_link_directories(A4_ImageExtractor PUBLIC "${MY_ZLIB_DIR}/lib")
target_link_libraries(A4_ImageExtractor cnpy zlib)

# A5_SingleToothExtractor
set(SRCS_A5
    src/A5_SingleToothExtractor.cpp
)
add_executable(A5_SingleToothExtractor "${SRCS_A5}")
target_include_directories(A5_SingleToothExtractor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(A5_SingleToothExtractor PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl/include")
target_include_directories(A5_SingleToothExtractor PUBLIC "${GC_EIGEN_LOCATION}")
target_include_directories(A5_SingleToothExtractor PUBLIC "${MY_CNPY_DIR}/include")
target_include_directories(A5_SingleToothExtractor PUBLIC "${MY_ZLIB_DIR}/include")
target_link_directories(A5_SingleToothExtractor PUBLIC "${MY_CNPY_DIR}/lib")
target_link_directories(A5_SingleToothExtractor PUBLIC "${MY_ZLIB_DIR}/lib")
target_link_libraries(A5_SingleToothExtractor igl::core cnpy zlib)

# E1_RemapUV
set(SRCS_E1
    src/E1_RemapUV.cpp
)
add_executable(E1_RemapUV "${SRCS_E1}")
target_include_directories(E1_RemapUV PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(E1_RemapUV PUBLIC "${MY_CNPY_DIR}/include")
target_include_directories(E1_RemapUV PUBLIC "${MY_ZLIB_DIR}/include")
target_link_directories(E1_RemapUV PUBLIC "${MY_CNPY_DIR}/lib")
target_link_directories(E1_RemapUV PUBLIC "${MY_ZLIB_DIR}/lib")
target_link_libraries(E1_RemapUV geometry-central polyscope cnpy zlib)

# test
#find_package(ZLIB REQUIRED)
add_executable(TestProgram "src/test.cpp")
target_include_directories(TestProgram PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/libigl/include")
target_include_directories(TestProgram PUBLIC "${GC_EIGEN_LOCATION}")
target_include_directories(TestProgram PUBLIC "${MY_CNPY_DIR}/include")
target_include_directories(TestProgram PUBLIC "${MY_ZLIB_DIR}/include")
target_link_directories(TestProgram PUBLIC "${MY_CNPY_DIR}/lib")
target_link_directories(TestProgram PUBLIC "${MY_ZLIB_DIR}/lib")
target_link_libraries(TestProgram igl::core cnpy zlib)