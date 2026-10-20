cmake_minimum_required(VERSION 3.15)
project(NumericalMethods CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(num STATIC
    src/lu.cpp
    src/tridiagonal.cpp
    src/iterative.cpp
    src/jacobi.cpp
    src/qr.cpp
)
target_include_directories(num PUBLIC include)

add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE num)

if (MSVC)
    target_compile_options(num PUBLIC /utf-8)
    target_compile_options(app PRIVATE /utf-8)
endif()

file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})