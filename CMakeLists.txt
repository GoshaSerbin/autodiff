cmake_minimum_required(VERSION 3.5.0)
project(auto_diff
    VERSION 1.0 
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==========================================================
# Options
# ==========================================================
option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_COVERAGE "Enable coverage" ON)
option(ENABLE_SANITIZERS "Enable sanitizers" ON)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_OPENMP "Enable OpenMP support" OFF)
option(USE_ANDROID_LOGGING "Enable Android logging" OFF)

# ==========================================================
# Compile options
# ==========================================================
add_compile_options(-Wall -Wextra -Werror -Wpedantic)

# ==========================================================
# Source code
# ==========================================================
set(SOURCES
    src/main.cpp
)
set(INCLUDES
include
)

include_directories(${INCLUDES} ${PROJECT_BINARY_DIR})
add_executable(${PROJECT_NAME} ${SOURCES})

# ==========================================================
# Compile definitions
# ==========================================================
target_compile_definitions(${PROJECT_NAME} PRIVATE PROJECT_NAME="${PROJECT_NAME}")


# ==========================================================
# Logger
# ==========================================================
add_subdirectory(libs/Logger)
target_link_libraries(${PROJECT_NAME} PRIVATE Logger)

# ==========================================================
# AutoDiff
# ==========================================================
add_subdirectory(libs/AutoDiff)
target_link_libraries(${PROJECT_NAME} PRIVATE AutoDiff)


if (ENABLE_OPENMP)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
target_compile_definitions(${PROJECT_NAME} PUBLIC USE_OPENMP)
endif()
endif()

if (ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # add_library(cuda_code OBJECT cuda_code.cu)
    # set_target_properties(cuda_code PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    target_link_libraries(${PROJECT_NAME} PRIVATE cuda_code)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CUDA)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    message(STATUS "Configuring for Android")
    

    if(USE_ANDROID_LOGGING)
        add_compile_definitions(USE_ANDROID_LOGGING)
        target_link_libraries(${PROJECT_NAME} PRIVATE log)
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANDROID")

    # Задаем ABI и платформу (можно переопределить при вызове CMake)
    set(ANDROID_ABI "arm64-v8a" CACHE STRING "Android ABI")
    set(ANDROID_PLATFORM 24 CACHE STRING "Android API Level")

    # Опции компиляции для Android
    target_compile_options(${PROJECT_NAME} PRIVATE -fPIC)

endif()

if(ENABLE_COVERAGE OR ENABLE_SANITIZERS OR ENABLE_TESTS)
    add_compile_options(-g -O0)
elseif()
    add_compile_options(-Ofast)
endif()

if(ENABLE_COVERAGE)
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

if(ENABLE_SANITIZERS)
    add_compile_options(-fsanitize=address -fsanitize=leak -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=leak -fsanitize=undefined)
endif()

if(ENABLE_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_subdirectory(libs/AutoDiff/tests)
    add_subdirectory(libs/Logger/tests)
endif()

