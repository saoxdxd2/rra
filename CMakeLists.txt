cmake_minimum_required(VERSION 3.20)
cmake_policy(VERSION 3.20)
project(rra_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# Gracefully handle OpenCV paths across operating systems
if(WIN32 AND NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR "C:/opencv/build" CACHE PATH "OpenCV directory")
endif()
find_package(OpenCV REQUIRED)

# Modern approach to OpenMP
find_package(OpenMP REQUIRED)

include(FetchContent)

# Nlohmann JSON (Manual)
add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/_deps/json-src/include)

# cpp-httplib (Manual)
add_library(httplib INTERFACE)
add_library(httplib::httplib ALIAS httplib)
target_include_directories(httplib INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/_deps/httplib-src)

# Commented out FetchContent to avoid network failures
# FetchContent_Declare(json ...)
# FetchContent_Declare(httplib ...)

# Consolidate shared sources to remove redundant logic
set(COMMON_SOURCES
    src/core_math.cpp
    src/nis_engine.cpp
    src/byte_field.cpp
    src/tokenizer.cpp
)

set(COMMON_HEADERS
    include/neural_types.hpp
    include/nis_engine.hpp
    include/neuron.hpp
    include/synapse.hpp
    include/tokenizer.hpp
)

add_executable(launcher train.cpp ${COMMON_SOURCES} ${COMMON_HEADERS})
add_executable(generate generate.cpp ${COMMON_SOURCES} ${COMMON_HEADERS})
add_executable(manifold_ui src/manifold_ui.cpp)

# Apply includes, libraries, and compiler optimizations to ALL targets
set(ALL_TARGETS launcher generate manifold_ui)

foreach(TARGET_NAME IN LISTS ALL_TARGETS)
    target_include_directories(${TARGET_NAME} PRIVATE include . ${OpenCV_INCLUDE_DIRS})
    
    # Link OpenMP as a modern imported target rather than passing -fopenmp manually
    target_link_libraries(${TARGET_NAME} PRIVATE 
        ${OpenCV_LIBS} 
        nlohmann_json::nlohmann_json 
        httplib::httplib
        OpenMP::OpenMP_CXX
    )

    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE
            /Ox
            /arch:AVX512
            /fp:fast
            /favor:INTEL64
            /EHsc
            /DNDEBUG
            /wd4805
            /GL
        )
        target_link_options(${TARGET_NAME} PRIVATE /LTCG)
    else()
        # -march=native automatically unlocks AVX-512 optimizations for the host CPU
        target_compile_options(${TARGET_NAME} PRIVATE
            -O3
            -ffast-math
            -march=native
            -mfma
            -flto
        )
    endif()
endforeach()