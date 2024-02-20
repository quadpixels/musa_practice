cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME musa_practice)
project(${PROJECT_NAME} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
find_package(MUSA REQUIRED)

musa_add_executable(deviceProperties deviceProperties.mu)
target_link_libraries(deviceProperties musa musart)
target_link_directories(deviceProperties PUBLIC /usr/local/musa/lib)

musa_add_executable(radixSort radixSort.mu)
target_link_libraries(radixSort musa musart)
target_link_directories(radixSort PUBLIC /usr/local/musa/lib)