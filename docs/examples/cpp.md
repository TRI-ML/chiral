# C++ Examples

Both programs are complete, compilable source files. Build and run them as follows:

```bash
cmake -S examples/cpp -B build_ex -DCMAKE_BUILD_TYPE=Release
cmake --build build_ex --parallel
./build_ex/server_example &
./build_ex/client_example
```

---

## Server Example

**File:** `examples/cpp/server_example.cpp`

This program implements a `MyRobotServer` subclass that simulates four cameras with depth, two proprio streams, and updating extrinsics (simulating wrist cameras oscillating).

### Includes and Type Aliases

```cpp
#include <chiral/server.hpp>
#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;
```

`<chiral/server.hpp>` is the only Chiral header needed on the server side. It transitively includes `types.hpp` for `CameraConfig`, `ProprioConfig`, `Observation`, `Action`, `StepResult`, and `InfoMap`. `<Eigen/Dense>` is pulled in by `types.hpp` since `CameraConfig` uses `Eigen::Matrix3d` / `Eigen::Matrix4d`.

The `Clock` and `Ms` aliases are for measuring step timing.

### Camera Configuration Function

```cpp
static constexpr int H   = 480;
static constexpr int W   = 640;
static constexpr int DOF = 7;

static std::vector<chiral::CameraConfig> make_cam_configs() {
    Eigen::Matrix3d K;
    K << 600,   0, 320,
           0, 600, 240,
           0,   0,   1;

    std::vector<chiral::CameraConfig> cfgs;
    for (const char* name : {"cam_0", "cam_1", "cam_2", "cam_3"}) {
        chiral::CameraConfig c;
        c.name      = name;
        c.height    = H;
        c.width     = W;
        c.channels  = 3;
        c.has_depth = true;
        c.intrinsics = K;
        c.extrinsics = Eigen::Matrix4d::Identity();
        cfgs.push_back(c);
    }
    return cfgs;
}
```

`make_cam_configs()` is a free function that builds the camera config vector. The `<<` syntax sets all nine elements of the `Eigen::Matrix3d` in row-major order — this is the pinhole camera matrix K with `fx=fy=600`, principal point `(320, 240)`.

All four cameras share the same K (a simplification; real systems have per-camera calibration). The extrinsics start as the 4×4 identity and will be updated in the camera loop.

`has_depth = true` tells the server base class to allocate a `H*W*sizeof(float)` byte buffer for each camera's depth map.

### Proprio Configuration Function

```cpp
static std::vector<chiral::ProprioConfig> make_proprio_configs() {
    return {{"joint_pos", DOF}, {"joint_vel", DOF}};
}
```

Two streams, each of length 7. The `ProprioConfig` struct supports aggregate initialisation, so `{"joint_pos", DOF}` is equivalent to `ProprioConfig{.name = "joint_pos", .size = DOF}`.

### Class Definition and Constructor

```cpp
class MyRobotServer : public chiral::PolicyServer {
    int               step_    = 0;
    double            t_sum_   = 0.0;
    std::atomic<bool> running_{true};

public:
    MyRobotServer(const std::string& host, int port)
        : PolicyServer(make_cam_configs(), make_proprio_configs(), host, port)
    {
        // One capture thread per camera.
        for (std::size_t i = 0; i < configs_.size(); ++i)
            std::thread(&MyRobotServer::camera_loop, this, i).detach();

        // One update thread per proprio stream.
        for (std::size_t i = 0; i < proprio_configs_.size(); ++i)
            std::thread(&MyRobotServer::proprio_loop, this, i).detach();
    }

    ~MyRobotServer() { running_ = false; }
```

The member initializer `: PolicyServer(make_cam_configs(), make_proprio_configs(), host, port)` calls the base class constructor, which pre-allocates `images_`, `depths_`, `intrinsics_`, `extrinsics_`, `proprios_`, and all mutexes. Only after this returns are `configs_` and `proprio_configs_` valid to read.

`running_` is an `std::atomic<bool>` to allow safe reads from camera and proprio threads alongside the write in the destructor.

`std::thread(...).detach()` launches each thread and releases ownership — the threads run independently until `running_` becomes `false`. The destructor sets `running_ = false` and then returns, at which point all threads will exit their loops at the next `sleep` call.

!!! note "Thread lifetime"
    Detached threads continue running until they observe `running_ == false`. Because the sleep interval is at most 33 ms (camera) or 2 ms (proprio), all threads exit within one sleep interval after the destructor runs.

### get_metadata

```cpp
    chiral::InfoMap get_metadata() override {
        return {{"cameras", "cam_0,cam_1,cam_2,cam_3"}, {"action_N", "1"}, {"action_D", "7"}};
    }
```

`InfoMap` is `unordered_map<string, string>`. All values must be strings. The camera list is serialized as a comma-separated string — the client can split on `,` to get individual names. Numeric values (`action_N`, `action_D`) are stored as decimal strings and parsed with `std::stoi` on the client side.

### reset

```cpp
    std::pair<chiral::Observation, chiral::InfoMap> reset() override {
        step_ = 0; t_sum_ = 0.0;
        return {make_obs(), {}};
    }
```

`make_obs()` (no arguments) takes a snapshot at timestamp `0.0`. The second element of the pair is an empty `InfoMap`. In a real system, add a command to move the robot to its home configuration before this call.

### step

```cpp
    chiral::StepResult step(const chiral::Action& /*action*/) override {
        auto t0 = Clock::now();

        // make_obs() snapshots all camera and proprio buffers under their locks.
        chiral::StepResult r;
        r.obs       = make_obs(step_ * 0.05);
        r.reward    = 0.f;
        r.truncated = false;

        double step_ms = Ms(Clock::now() - t0).count();
        t_sum_ += step_ms;
        r.terminated = ++step_ >= 200;

        if (step_ % 10 == 0)
            std::printf("step=%4d  server_step=%5.2fms  avg=%5.2fms\n",
                        step_, step_ms, t_sum_ / step_);
        return r;
    }
```

`make_obs(step_ * 0.05)` snapshots all four cameras and both proprio streams. Each camera's snapshot acquires its mutex, copies the four buffers (image, depth, intrinsics, extrinsics), and releases the mutex. The step timing measures how long the snapshot itself takes — typically well under 1 ms for in-memory copies.

The `action` parameter is ignored in this example (comment `/*action*/`). In a real system, decode and apply it to the robot here.

### Camera Loop

```cpp
private:
    void camera_loop(std::size_t idx) {
        // Simulates a sensor driver running at ~30 Hz.
        double t = 0.0;
        while (running_) {
            // ── hardware capture goes here ─────────────────────────────────
            // update_image(idx, new_image.data(), new_image.size());
            // update_depth(idx, new_depth.data(), new_depth.size());
            // ──────────────────────────────────────────────────────────────

            // Update extrinsics every frame for moving cameras (e.g. wrist).
            // In real code: T = fk_solver.compute(joint_positions)
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T(0, 3) = 0.1 * std::sin(t);  // simulated oscillating translation
            update_extrinsics(idx, T);

            t += 1.0 / 30.0;
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }
```

`update_image(idx, ptr, len)` and `update_depth(idx, ptr, len)` take a raw `uint8_t*` and byte length. For image data, `len = H * W * channels`. For depth data, `len = H * W * sizeof(float)`.

`update_extrinsics(idx, T)` accepts an `Eigen::Matrix4d`. Internally it acquires `cam_mutexes_[idx]`, copies the 16 doubles into `extrinsics_[idx]`, and releases the mutex.

`idx` is the zero-based index into `configs_` — not the camera name. Camera thread `i` always operates on camera `i`.

### Proprio Loop

```cpp
    void proprio_loop(std::size_t idx) {
        // Simulates a proprioception driver running at ~500 Hz.
        const int n = proprio_configs_[idx].size;
        Eigen::VectorXf state = Eigen::VectorXf::Zero(n);
        while (running_) {
            // ── hardware read goes here ────────────────────────────────────
            // state = robot.read_joints();  (Eigen::VectorXf, length n)
            update_proprio(idx, state.data(), state.size());
            // ──────────────────────────────────────────────────────────────
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
```

`update_proprio(idx, ptr, count)` takes a `float*` and element count (not byte count). `Eigen::VectorXf::data()` returns a `float*` and `size()` returns the element count — they compose naturally.

The state is zero-initialized and never updated in this example. In a real system, `robot.read_joints()` would return the actual joint positions.

### main

```cpp
int main() {
    MyRobotServer("0.0.0.0", 8765).run();
}
```

`run()` is blocking. The server listens on all interfaces (`0.0.0.0`) at port `8765` until the process is killed.

---

## Client Example

**File:** `examples/cpp/client_example.cpp`

The client connects, fetches metadata, resets, and runs the episode loop while printing per-step timing and camera summaries.

### Includes and Aliases

```cpp
#include <chiral/client.hpp>
#include <Eigen/Dense>
#include <chrono>
#include <cstdio>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;
using Sec   = std::chrono::duration<double>;
```

`<chiral/client.hpp>` pulls in `PolicyClient` and all types from `types.hpp`.

### Connection and Metadata

```cpp
int main() {
    chiral::PolicyClient env("ws://localhost:8765");
    env.connect();

    auto meta = env.get_metadata();
    std::printf("metadata:\n");
    for (const auto& kv : meta)
        std::printf("  %s: %s\n", kv.first.c_str(), kv.second.c_str());
    std::printf("\n");
```

`connect()` polls until the WebSocket handshake succeeds. After 500 ms of failure it prints a waiting message, then continues polling at 100 ms intervals. Once connected it returns a reference to `env`.

`get_metadata()` sends a `metadata` request and returns an `InfoMap`. The range-based for loop prints all key-value pairs — all values are strings regardless of the Python type on the server side.

### Reset

```cpp
    auto [obs, info] = env.reset();
```

C++17 structured bindings decompose the `pair<Observation, InfoMap>` return value into `obs` and `info`. The observation contains all camera images, depths, intrinsics, extrinsics, and proprio data from the server's current state.

### Episode Variables

```cpp
    float total_reward = 0.f;
    int   step         = 0;
    auto  t_episode    = Clock::now();
```

### Episode Loop

```cpp
    while (true) {
        // Build action with Eigen, then hand off to the API.
        Eigen::Matrix<float, 1, 7> action_mat = Eigen::Matrix<float, 1, 7>::Zero();
        // policy_net.forward(obs) → action_mat   (replace with real inference)

        chiral::Action action;
        action.N = 1; action.D = 7;
        action.data.assign(action_mat.data(), action_mat.data() + action_mat.size());
```

The action is built as an Eigen matrix first (where policy inference would happen), then copied into a `chiral::Action`. `action_mat.data()` returns a `float*` to the contiguous row-major storage; `assign` does one memcpy.

!!! tip "Avoid extra allocations"
    For performance in tight loops, pre-allocate `action.data` outside the loop with `action.data.resize(N * D)` and then use `std::copy` instead of `assign`. `assign` reallocates each iteration if the size doesn't match capacity.

### Stepping

```cpp
        auto   t0      = Clock::now();
        auto   res     = env.step(action);
        double latency = Ms(Clock::now() - t0).count();

        obs           = std::move(res.obs);
        total_reward += res.reward;
        ++step;
```

`env.step(action)` sends the action, waits for the server's `step_response`, and returns a `StepResult`. `std::move(res.obs)` transfers ownership of the observation buffers without copying.

### First-step Camera Summary

```cpp
        // Print a full summary on the first step, then just timing stats.
        if (step == 1) {
            std::printf("cameras:\n");
            for (const auto& cam : obs.cameras) {
                // cam.intrinsics is Eigen::Matrix3d — extract K components directly.
                double fx = cam.intrinsics(0, 0), fy = cam.intrinsics(1, 1);
                double cx = cam.intrinsics(0, 2), cy = cam.intrinsics(1, 2);

                // cam.extrinsics is Eigen::Matrix4d — translation = last column.
                Eigen::Vector3d pos = cam.extrinsics.col(3).head<3>();

                std::printf("  %s(%dx%d)  fx=%.0f fy=%.0f cx=%.0f cy=%.0f"
                            "  pos=[%.2f %.2f %.2f]\n",
                            cam.name.c_str(),
                            cam.image_shape[0], cam.image_shape[1],
                            fx, fy, cx, cy,
                            pos.x(), pos.y(), pos.z());
            }
```

On the first step the code iterates `obs.cameras` and prints the intrinsics and extrinsics for each camera. Two patterns are shown:

- `cam.intrinsics(0, 0)` — Eigen matrix element access with row/column indices.
- `cam.extrinsics.col(3).head<3>()` — extract the translation vector from the last column of the 4×4 homogeneous matrix. This is the camera position in world coordinates.

!!! note "Intrinsics and extrinsics are fresh every step"
    Even though this summary is printed only on step 1, `cam.intrinsics` and `cam.extrinsics` are always the values from the **current** step. For a wrist camera, `pos` changes every step as the arm moves.

### Proprio Summary

```cpp
            if (!obs.proprios.empty()) {
                std::printf("proprios:\n");
                for (const auto& p : obs.proprios) {
                    // Map the raw float buffer as an Eigen vector — zero-copy.
                    Eigen::Map<const Eigen::VectorXf> v(p.data.data(), p.data.size());
                    std::printf("  %s[%zu]  norm=%.4f\n",
                                p.name.c_str(), p.data.size(), v.norm());
                }
            }
```

`Eigen::Map<const Eigen::VectorXf>` wraps the `vector<float>` as a read-only Eigen vector without copying. This gives access to all Eigen operations (`norm()`, `dot()`, etc.) directly on the received data.

### Per-step Timing

```cpp
        std::printf("step=%4d  latency=%6.1fms  fps=%6.1f\n", step, latency, fps);
```

### Termination Check

```cpp
        if (res.terminated || res.truncated) {
            std::printf("\ndone — steps=%d  total_reward=%.2f  avg_fps=%.1f\n",
                        step, total_reward, fps);
            break;
        }
    }

    env.close();
}
```

`env.close()` closes the WebSocket connection and frees resources. This is also called by the destructor if you don't call `close()` explicitly, but explicit cleanup makes intentions clear.

---

## Expected Output

**Server terminal:**
```
step=  10  server_step= 0.04ms  avg= 0.04ms
step=  20  server_step= 0.04ms  avg= 0.04ms
...
step= 200  server_step= 0.04ms  avg= 0.04ms
```

**Client terminal:**
```
metadata:
  cameras: cam_0,cam_1,cam_2,cam_3
  action_N: 1
  action_D: 7

cameras:
  cam_0(480x640)  fx=600 fy=600 cx=320 cy=240  pos=[0.10 0.00 0.00]
  cam_1(480x640)  fx=600 fy=600 cx=320 cy=240  pos=[0.10 0.00 0.00]
  cam_2(480x640)  fx=600 fy=600 cx=320 cy=240  pos=[0.10 0.00 0.00]
  cam_3(480x640)  fx=600 fy=600 cx=320 cy=240  pos=[0.10 0.00 0.00]
proprios:
  joint_pos[7]  norm=0.0000
  joint_vel[7]  norm=0.0000

step=   1  latency=  40.2ms  fps=  24.4
step=   2  latency=  38.7ms  fps=  25.1
...

done — steps=200  total_reward=0.00  avg_fps=24.8
```

!!! tip "C++ vs Python latency"
    The C++ client typically shows slightly lower latency than the Python client for the same server, since msgpack decoding and buffer reconstruction in C++ avoids Python's per-object allocation overhead.

---

## Adapting for a Real Robot

To connect the server to real hardware, modify the following in `server_example.cpp`:

1. **`camera_loop`**: Uncomment `update_image` / `update_depth` and replace with actual camera SDK calls. The data pointer and byte length must match the configured `H * W * channels` (image) and `H * W * sizeof(float)` (depth).

2. **`camera_loop`**: Replace the simulated `T(0,3) = 0.1 * sin(t)` with a call to your FK solver:
   ```cpp
   Eigen::VectorXf joints = robot.read_joints();
   Eigen::Matrix4d T = fk_solver.compute(joints, configs_[idx].name);
   update_extrinsics(idx, T);
   ```

3. **`proprio_loop`**: Replace the zero state with actual hardware reads:
   ```cpp
   Eigen::VectorXf state = robot.read_joints();  // length = proprio_configs_[idx].size
   update_proprio(idx, state.data(), state.size());
   ```

4. **`step`**: Apply the received action to the robot before calling `make_obs()`:
   ```cpp
   Eigen::Map<const Eigen::VectorXf> cmd(action.data.data(), action.D);
   robot.apply_joint_command(cmd);
   ```

5. **`reset`**: Command the robot to its home configuration and wait for it to settle before calling `make_obs()`.

In `client_example.cpp`:

1. Replace `Eigen::Matrix<float,1,7>::Zero()` with actual policy inference.
2. Parse `meta.at("action_N")` and `meta.at("action_D")` to get the correct `Action` dimensions from the server at runtime rather than hardcoding `N=1, D=7`.
