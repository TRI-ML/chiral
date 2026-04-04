# Client Guide

`PolicyClient` is the policy-side entry point. It connects over WebSocket (or Zenoh) to a running `PolicyServer`. Observations and actions flow on independent background threads so chunked policy predictions never block waiting for camera data. All network I/O is handled internally.

---

## Connecting

### Python

The recommended pattern is the context manager, which calls `connect()` on entry and `close()` on exit — even if an exception is raised inside the block.

```python
import chiral

with chiral.PolicyClient("ws://localhost:8765") as env:
    obs, info = env.reset()
    # ... episode loop ...
```

For longer-lived usage (e.g. when the client is a member of a larger object) use explicit `connect` / `close`:

```python
env = chiral.PolicyClient("ws://robot-host:8765")
env.connect()

try:
    obs, info = env.reset()
    # ... episode loop ...
finally:
    env.close()
```

### C++

C++ uses explicit `connect` / `close`:

```cpp
#include <chiral/client.hpp>

chiral::PolicyClient env("ws://localhost:8765");
env.connect();

auto [obs, info] = env.reset();
// ... episode loop ...

env.close();
```

`connect()` returns a reference to the client so you can chain:

```cpp
chiral::PolicyClient env("ws://localhost:8765");
auto [obs, info] = env.connect().reset();
```

### Alternative: Zenoh transport

Pass `protocol="zenoh"` to use Zenoh over TCP instead of WebSocket. The rest of the API is identical.

```python
with chiral.PolicyClient("tcp/localhost:7447", protocol="zenoh") as env:
    obs, info = env.reset()
    ...
```

The default URI when `protocol="zenoh"` is `"tcp/localhost:7447"`, so it can be omitted when connecting to localhost:

```python
with chiral.PolicyClient(protocol="zenoh") as env:
    ...
```

### Polling Behavior

`connect()` polls the server until the connection is established. For WebSocket it retries every 100 ms; for Zenoh it sends a probe metadata request every 500 ms. If the server is not yet available a waiting message is printed:

```
chiral: waiting for server at ws://localhost:8765 …
```

This means you can start the client before the server is ready — it will block until the connection opens.

!!! tip "Starting server and client together"
    ```bash
    python server.py &
    python client.py   # will wait automatically if server isn't ready yet
    ```

---

## Getting Metadata

`get_metadata()` sends a request to the server and returns whatever the server's `get_metadata()` override returns. Call it once after connecting, before `reset()`.

=== "Python"

    ```python
    meta = env.get_metadata()
    # meta is a plain dict; content is application-defined.
    cameras      = meta.get("cameras", [])       # e.g. ["wrist_cam", "head_cam"]
    action_shape = meta.get("action_shape", [1, 7])
    print(f"cameras: {cameras}  action_shape: {action_shape}")
    ```

=== "C++"

    ```cpp
    chiral::InfoMap meta = env.get_metadata();
    // InfoMap = unordered_map<string, string>
    // All values are strings; parse as needed.
    for (const auto& [key, val] : meta)
        std::printf("  %s: %s\n", key.c_str(), val.c_str());

    // Parse a numeric value:
    int action_D = std::stoi(meta.at("action_D"));
    ```

!!! note "InfoMap is string-only in C++"
    The C++ `InfoMap` type is `unordered_map<string, string>`. If the server stores non-string metadata (e.g. a Python int or list), it is automatically stringified when the response reaches a C++ client. Parse it back with `std::stoi`, `std::stof`, or a JSON/CSV parser as appropriate.

---

## Episode Loop

The typical pattern uses three concurrent threads:

1. **Obs stream** (`start_obs_stream`) — polls the server at a fixed Hz and stores the latest observation in `latest_obs`.
2. **Policy loop** (your code) — reads `latest_obs`, runs inference, and enqueues a chunk of actions with `put_action`.
3. **Action dispatch** (`start_action_dispatch`) — dequeues one action at a time and sends it to the server at a fixed Hz.

Because `apply_action` on the server is fire-and-forget (no response), the obs stream is never blocked by action sending.

=== "Python"

    ```python
    import threading
    import numpy as np
    import chiral

    CHUNK_SIZE = 10  # actions predicted per inference call

    def policy_loop(env, stop):
        while not stop.is_set():
            obs = env.latest_obs
            if obs is None:
                continue
            # Replace with real model inference:
            actions = np.zeros([CHUNK_SIZE, 7], dtype=np.float32)
            for a in actions:
                env.put_action(a)

    with chiral.PolicyClient("ws://localhost:8765") as env:
        meta = env.get_metadata()
        obs, info = env.reset()

        env.start_obs_stream(hz=30)       # fetch obs at 30 Hz
        env.start_action_dispatch(hz=10)  # dispatch one action per tick at 10 Hz

        stop = threading.Event()
        t = threading.Thread(target=policy_loop, args=(env, stop))
        t.start()

        # Run for desired duration, then stop.
        stop.set(); t.join()
    ```

=== "C++"

    > **Note:** The C++ client still uses the legacy coupled `step()` API.

    ```cpp
    #include <chiral/client.hpp>
    #include <cstdio>

    int main() {
        chiral::PolicyClient env("ws://localhost:8765");
        env.connect();

        auto meta = env.get_metadata();
        int N = std::stoi(meta.count("action_N") ? meta.at("action_N") : "1");
        int D = std::stoi(meta.count("action_D") ? meta.at("action_D") : "7");

        auto [obs, info] = env.reset();

        int step = 0;

        while (true) {
            // Run your policy here.  Replace with real model inference.
            chiral::Action action;
            action.N = N;
            action.D = D;
            action.data.assign(N * D, 0.f);

            auto res = env.step(action);
            obs = std::move(res.obs);
            ++step;

            if (res.terminated || res.truncated) {
                std::printf("Episode done: steps=%d\n", step);
                break;
            }
        }

        env.close();
    }
    ```

---

## Accessing Observations

Every `reset()` and `step()` returns an `Observation`. This section shows how to access every field.

### Camera Data by Name

Look up a camera by name using `obs["name"]`. This performs a linear search through `obs.cameras` and throws `KeyError` (Python) or `std::out_of_range` (C++) if the name is not found.

=== "Python"

    ```python
    cam = obs["wrist_cam"]   # returns CameraInfo; raises KeyError if absent

    # Pixel data
    image = cam.image        # np.ndarray (H, W, C) uint8

    # Depth map — present only if the server declared has_depth=True for this camera
    if cam.depth is not None:
        depth = cam.depth    # np.ndarray (H, W) float32, values in metres

    # Camera matrix K — (3, 3) float64 — FRESH EVERY STEP
    K  = cam.intrinsics
    fx = K[0, 0];  fy = K[1, 1]
    cx = K[0, 2];  cy = K[1, 2]

    # Camera-to-world transform T — (4, 4) float64 — FRESH EVERY STEP
    T = cam.extrinsics
    R = T[:3, :3]    # rotation matrix (3, 3)
    t = T[:3,  3]    # camera position in world frame (3,)
    ```

=== "C++"

    ```cpp
    const chiral::CameraInfo& cam = obs["wrist_cam"];  // throws std::out_of_range if absent

    // Pixel data — raw bytes, shape is in cam.image_shape
    const std::vector<uint8_t>& image = cam.image;
    int H = cam.image_shape[0];
    int W = cam.image_shape[1];
    int C = cam.image_shape[2];

    // Depth map — only populated if server declared has_depth=true
    if (cam.has_depth) {
        const std::vector<uint8_t>& depth_bytes = cam.depth_data;
        // Reinterpret as float32:
        const float* depth_ptr =
            reinterpret_cast<const float*>(depth_bytes.data());
        // depth_ptr[row * W + col] gives metres at (row, col)
    }

    // Camera matrix K — Eigen::Matrix3d — FRESH EVERY STEP
    const Eigen::Matrix3d& K = cam.intrinsics;
    double fx = K(0, 0),  fy = K(1, 1);
    double cx = K(0, 2),  cy = K(1, 2);

    // Camera-to-world transform T — Eigen::Matrix4d — FRESH EVERY STEP
    const Eigen::Matrix4d& T = cam.extrinsics;
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);   // rotation
    Eigen::Vector3d t = T.col(3).head<3>();      // camera position in world frame
    ```

!!! warning "Intrinsics and extrinsics are fresh every step"
    Unlike many robotics frameworks that send camera calibration only at startup, Chiral includes the full K and T matrices in **every** observation. Do not cache intrinsics or extrinsics from a previous step — always read them from the current observation. This is the correct behavior for wrist cameras, head cameras, or any camera whose pose changes during an episode.

### Iterating All Cameras

When you need to process all cameras without knowing their names in advance:

=== "Python"

    ```python
    for cam in obs.cameras:
        print(f"{cam.name}: image={cam.image.shape}  "
              f"K[fx,fy]=[{cam.intrinsics[0,0]:.1f},{cam.intrinsics[1,1]:.1f}]  "
              f"depth={'yes' if cam.depth is not None else 'no'}")
    ```

=== "C++"

    ```cpp
    for (const auto& cam : obs.cameras) {
        std::printf("%s: image=[%d,%d,%d]  fx=%.1f fy=%.1f  depth=%s\n",
                    cam.name.c_str(),
                    cam.image_shape[0], cam.image_shape[1], cam.image_shape[2],
                    cam.intrinsics(0,0), cam.intrinsics(1,1),
                    cam.has_depth ? "yes" : "no");
    }
    ```

### Proprioception

Access proprio streams by name via `obs.proprios`:

=== "Python"

    ```python
    # obs.proprios is a dict[str, np.ndarray], each value a 1-D float32 array.
    joint_pos = obs.proprios["joint_pos"]   # np.ndarray (DOF,) float32
    joint_vel = obs.proprios["joint_vel"]

    # Iterate all streams:
    for name, values in obs.proprios.items():
        print(f"  {name}: {values}")
    ```

=== "C++"

    ```cpp
    // obs.proprios is a std::vector<ProprioInfo>.
    // Use obs.proprio(name) to look up by name (throws std::out_of_range if absent).
    const chiral::ProprioInfo& jp = obs.proprio("joint_pos");
    const std::vector<float>& joint_pos = jp.data;  // length DOF

    // Zero-copy Eigen map over the raw float buffer:
    Eigen::Map<const Eigen::VectorXf> q(joint_pos.data(), joint_pos.size());
    std::printf("joint_pos norm = %.4f\n", q.norm());

    // Iterate all streams:
    for (const auto& p : obs.proprios)
        std::printf("  %s[%zu]\n", p.name.c_str(), p.data.size());
    ```

### Timestamp

```python
# Python
print(obs.timestamp)   # float, application-defined (often step * dt or wall time)
```

```cpp
// C++
std::printf("timestamp = %.4f\n", obs.timestamp);
```

---

## Building Actions

Actions are 2-D float32 tensors of shape `[N, D]` where `N` is the number of action steps in a chunk and `D` is the dimensionality (e.g. 7 for a 7-DOF arm). Single-step policies use `N=1`.

=== "Python"

    The client accepts any array-like that can be cast to float32. The most common pattern:

    ```python
    import numpy as np

    # Zero action (for testing)
    action = np.zeros([1, 7], dtype=np.float32)

    # From a policy network output (PyTorch example)
    # action = model(obs_tensor).detach().cpu().numpy()  # shape [1, 7]

    # Action chunk (N > 1, e.g. diffusion policy)
    action = np.zeros([8, 7], dtype=np.float32)

    obs, reward, terminated, truncated, info = env.step(action)
    ```

    !!! tip "dtype coercion"
        `encode_action` calls `np.asarray(action, dtype=np.float32)` internally, so you can pass an int array or a torch tensor — it will be cast to float32 before sending. However, be explicit for clarity.

=== "C++"

    Construct a `chiral::Action` manually or from an Eigen matrix:

    ```cpp
    // Manual construction
    chiral::Action action;
    action.N = 1;
    action.D = 7;
    action.data.assign(7, 0.f);  // N*D zeros

    // From an Eigen matrix (zero-copy after assign)
    Eigen::Matrix<float, 1, 7> mat = Eigen::Matrix<float, 1, 7>::Zero();
    // mat = policy_net.forward(obs_tensor);  // replace with real inference

    chiral::Action action;
    action.N = 1;
    action.D = 7;
    action.data.assign(mat.data(), mat.data() + mat.size());

    auto res = env.step(action);
    ```

    The `action.data` vector must contain exactly `N * D` float32 elements in row-major order. The server receives shape `[N, D]` and can reshape accordingly.

---

## Complete Client Example

=== "Python"

    ```python
    import time
    import numpy as np
    import chiral

    def run_episode(uri: str = "ws://localhost:8765"):
        with chiral.PolicyClient(uri) as env:
            # Fetch server metadata once.
            meta = env.get_metadata()
            cameras      = meta.get("cameras", [])
            action_shape = meta.get("action_shape", [1, 7])
            print(f"Server metadata: cameras={cameras}  action_shape={action_shape}")

            obs, info = env.reset()
            print(f"Reset: {len(obs.cameras)} cameras  timestamp={obs.timestamp}")

            total_reward = 0.0
            step = 0
            t_start = time.perf_counter()

            while True:
                # Read observation fields.
                cam = obs["wrist_cam"]
                K   = cam.intrinsics    # (3, 3) float64 — fresh this step
                T   = cam.extrinsics    # (4, 4) float64 — fresh this step

                # Run policy (replace with real inference).
                action = np.zeros(action_shape, dtype=np.float32)

                t0 = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                latency_ms = (time.perf_counter() - t0) * 1e3

                step += 1
                total_reward += reward
                fps = step / (time.perf_counter() - t_start)

                print(f"step={step:4d}  latency={latency_ms:5.1f}ms  "
                      f"fps={fps:5.1f}  reward={reward:.3f}")

                if terminated or truncated:
                    break

            print(f"\nDone — steps={step}  total_reward={total_reward:.2f}  "
                  f"avg_fps={fps:.1f}")

    if __name__ == "__main__":
        run_episode()
    ```

=== "C++"

    ```cpp
    #include <chiral/client.hpp>
    #include <Eigen/Dense>
    #include <chrono>
    #include <cstdio>

    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::duration<double, std::milli>;
    using Sec   = std::chrono::duration<double>;

    int main() {
        chiral::PolicyClient env("ws://localhost:8765");
        env.connect();

        // Fetch server metadata.
        auto meta = env.get_metadata();
        int N = meta.count("action_N") ? std::stoi(meta.at("action_N")) : 1;
        int D = meta.count("action_D") ? std::stoi(meta.at("action_D")) : 7;
        std::printf("Server: action=[%d,%d]\n\n", N, D);

        auto [obs, info] = env.reset();
        std::printf("Reset: %zu camera(s)  timestamp=%.4f\n",
                    obs.cameras.size(), obs.timestamp);

        float total_reward = 0.f;
        int   step         = 0;
        auto  t_start      = Clock::now();

        while (true) {
            // Read observation fields.
            const auto& cam = obs["wrist_cam"];
            // cam.intrinsics — Eigen::Matrix3d — FRESH THIS STEP
            // cam.extrinsics — Eigen::Matrix4d — FRESH THIS STEP
            double fx = cam.intrinsics(0, 0);
            Eigen::Vector3d pos = cam.extrinsics.col(3).head<3>();

            // Build action (replace with real inference).
            chiral::Action action;
            action.N = N; action.D = D;
            action.data.assign(N * D, 0.f);

            auto   t0      = Clock::now();
            auto   res     = env.step(action);
            double latency = Ms(Clock::now() - t0).count();

            obs           = std::move(res.obs);
            total_reward += res.reward;
            ++step;
            double fps = step / Sec(Clock::now() - t_start).count();

            std::printf("step=%4d  latency=%5.1fms  fps=%5.1f  reward=%.3f  "
                        "fx=%.0f  pos=[%.2f %.2f %.2f]\n",
                        step, latency, fps, res.reward,
                        fx, pos.x(), pos.y(), pos.z());

            if (res.terminated || res.truncated) {
                std::printf("\nDone — steps=%d  total_reward=%.2f  avg_fps=%.1f\n",
                            step, total_reward, fps);
                break;
            }
        }

        env.close();
    }
    ```
