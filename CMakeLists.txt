cmake_minimum_required(VERSION 3.18)
project(chiral VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# ── ixwebsocket ──────────────────────────────────────────────────────────────
FetchContent_Declare(ixwebsocket
    GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
    GIT_TAG        v11.4.5
    GIT_SHALLOW    TRUE
)
set(USE_TLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ixwebsocket)

# ── msgpack-cxx (header-only) ────────────────────────────────────────────────
FetchContent_Declare(msgpack
    GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
    GIT_TAG        cpp-6.1.1
    GIT_SHALLOW    TRUE
)
set(MSGPACK_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(MSGPACK_BUILD_TESTS       OFF CACHE BOOL "" FORCE)
set(MSGPACK_BUILD_DOCS        OFF CACHE BOOL "" FORCE)
set(MSGPACK_USE_BOOST         OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(msgpack)

# ── Eigen3 (header-only) ─────────────────────────────────────────────────────
# Prefer a system-installed Eigen; fall back to FetchContent.
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    FetchContent_Declare(eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    set(EIGEN_BUILD_DOC     OFF CACHE BOOL "" FORCE)
    set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING       OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(eigen3)
endif()

# ── chiral library ───────────────────────────────────────────────────────────
add_library(chiral
    src/protocol.cpp
    src/server.cpp
    src/client.cpp
)

target_include_directories(chiral PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ixwebsocket and msgpack-cxx are PRIVATE: they appear only in .cpp files,
# not in any installed header. Eigen3 is PUBLIC: types.hpp includes <Eigen/Dense>.
target_link_libraries(chiral
    PRIVATE ixwebsocket msgpack-cxx
    PUBLIC  Eigen3::Eigen
)

# ── install ──────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS chiral EXPORT chiralTargets)
install(DIRECTORY include/chiral DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT chiralTargets
    FILE      chiralTargets.cmake
    NAMESPACE chiral::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chiral
)

configure_package_config_file(
    cmake/chiralConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/chiralConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chiral
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/chiralConfigVersion.cmake"
    VERSION     ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/chiralConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/chiralConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/chiral
)
