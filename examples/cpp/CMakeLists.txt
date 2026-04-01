cmake_minimum_required(VERSION 3.18)
project(chiral_examples CXX)

set(CMAKE_CXX_STANDARD 14)

# Build chiral from source (adjust path if installed via find_package).
add_subdirectory(../.. chiral_build)

add_executable(server_example server_example.cpp)
target_link_libraries(server_example chiral)

add_executable(client_example client_example.cpp)
target_link_libraries(client_example chiral)
