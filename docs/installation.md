# Installation

---

## Python

### Requirements

| Requirement | Minimum version |
|-------------|----------------|
| Python | 3.10 |
| websockets | 12.0 |
| numpy | 1.24 |
| msgpack | 1.0 |
| eclipse-zenoh | 1.7.2 |
| tqdm | 4.0 |

### Install

From the repository root, install the package in editable mode with pip:

```bash
pip install .
```

Or non-editable:

```bash
pip install /path/to/chiral
```

### Development Extras

The `dev` extra adds `pytest` and `pytest-asyncio` for running the test suite. If you use [uv](https://github.com/astral-sh/uv):

```bash
uv sync --extra dev
```

With plain pip:

```bash
pip install ".[dev]"
```

### Verifying the Install

```python
import chiral
print(chiral.__version__)   # 0.1.0
```

If you can import `chiral` without errors, the install succeeded.

---

## C++

### Requirements

| Requirement | Minimum version | Notes |
|-------------|----------------|-------|
| CMake | 3.18 | Required for `FetchContent` |
| C++ compiler | C++14 | GCC, Clang, MSVC |
| ixwebsocket | v11.4.5 | Fetched automatically |
| msgpack-cxx | cpp-6.1.1 | Fetched automatically (header-only) |
| Eigen3 | 3.4 | System install preferred; fetched if absent |

### Automatic Dependencies

Chiral's `CMakeLists.txt` fetches three dependencies via `FetchContent`. You do not need to install them manually:

- **ixwebsocket v11.4.5** — WebSocket server/client implementation. TLS is disabled (`USE_TLS OFF`) to keep the dependency footprint minimal.
- **msgpack-cxx cpp-6.1.1** — Header-only C++ MessagePack library. Examples, tests, and Boost integration are all disabled.
- **Eigen 3.4** — Header-only linear algebra library used for `Matrix3d` (intrinsics) and `Matrix4d` (extrinsics) in the public API. CMake first tries `find_package(Eigen3 3.4 QUIET)`; if a sufficiently new system Eigen is found it is used as-is. If not, Eigen 3.4.0 is fetched from GitLab automatically.

!!! note "Transitive dependency"
    Only **Eigen3** is a transitive public dependency — it appears in the installed `chiral/types.hpp` header, so any downstream target that `#include`s Chiral headers also needs Eigen on its include path. `ixwebsocket` and `msgpack-cxx` are **private** build-time dependencies; they do not appear in any installed header and are not propagated to downstream targets.

### Build from Source

```bash
# Configure — Release mode recommended for benchmarks
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --parallel

# Optionally install to /usr/local (or a custom prefix)
cmake --install build
# or
cmake --install build --prefix /path/to/prefix
```

---

## Consuming Chiral from Another CMake Project

There are two ways to use Chiral in your own CMake project.

### Option A: add_subdirectory (simplest)

Clone or copy the Chiral source tree into your project (e.g. as a git submodule under `third_party/chiral`) and add it as a subdirectory:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_robot_project CXX)

add_subdirectory(third_party/chiral)

add_executable(my_policy_server src/main.cpp)
target_link_libraries(my_policy_server PRIVATE chiral)
```

CMake will automatically fetch ixwebsocket, msgpack-cxx, and Eigen (or use the system Eigen if available) when you configure your project.

!!! tip "Shallow submodule"
    When adding as a git submodule, use `--depth 1` or set `GIT_SHALLOW TRUE` in the `add_subdirectory` parent to keep clone size small:
    ```bash
    git submodule add --depth 1 https://github.com/tri-research/chiral third_party/chiral
    ```

### Option B: find_package after install

First install Chiral to a prefix (see above), then use CMake's `find_package` mechanism:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix /usr/local
```

In your downstream `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_robot_project CXX)

# Point CMake at the install prefix if it is not in a default search path
list(APPEND CMAKE_PREFIX_PATH /usr/local)

find_package(chiral REQUIRED)

add_executable(my_policy_server src/main.cpp)
target_link_libraries(my_policy_server PRIVATE chiral::chiral)
```

The installed package provides the namespaced target `chiral::chiral`. The `chiralConfig.cmake` and `chiralConfigVersion.cmake` files are installed alongside the library and satisfy version checking (`SameMajorVersion` compatibility).

!!! note "Eigen with find_package"
    When Chiral is installed and consumed via `find_package`, the Eigen3 dependency is propagated as a `PUBLIC` requirement. If Eigen3 is not on your system, CMake will report a missing dependency. Install it with your system package manager (`libeigen3-dev` on Ubuntu/Debian, `eigen` on Homebrew) or set `CMAKE_PREFIX_PATH` to point at an Eigen prefix.

---

## Running the Examples

After building, run the bundled examples to verify everything works:

```bash
# Python examples
uv run examples/python/server_example.py &
uv run examples/python/client_example.py

# C++ examples
cmake -S examples/cpp -B build_ex -DCMAKE_BUILD_TYPE=Release
cmake --build build_ex --parallel
./build_ex/server_example &
./build_ex/client_example
```

You should see per-step timing output on both server and client terminals.

!!! tip "Cross-language pair"
    Because Python and C++ share the same wire format, you can mix them freely:
    ```bash
    # Python server + C++ client
    uv run examples/python/server_example.py &
    ./build_ex/client_example

    # C++ server + Python client
    ./build_ex/server_example &
    uv run examples/python/client_example.py
    ```
