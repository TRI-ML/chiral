<img width="2048" height="467" alt="image (1)" src="https://github.com/user-attachments/assets/8156ace9-edc1-45dc-815d-3991903e1629" />

# Chiral

> **Alpha:** This project is in early alpha. APIs may change without notice.

Compact interface for robot policy evaluation. **[Documentation](https://tri-ml.github.io/chiral/)**

Sensor observations (images, depth maps, camera intrinsics/extrinsics, proprioception) are streamed from a robot **server** to a **policy client** in a separate process. Observations and actions flow on independent channels so chunked policy predictions never stall waiting for camera data.

```
  ┌─────────────────┐   obs stream (30 Hz)   ┌──────────────────┐
  │  PolicyServer   │ ─────────────────────► │  PolicyClient    │
  │  (robot side)   │                        │  (policy side)   │
  │                 │ ◄───────────────────── │                  │
  └─────────────────┘  action dispatch(10Hz) └──────────────────┘
```

One-to-one connection. No compression. Numpy arrays are transmitted as raw bytes using msgpack to minimize latency. Python and C++ share the same wire format, so cross-language pairs work.

**Transport:** WebSocket by default. Pass `protocol="zenoh"` to both server and client to use [Zenoh](https://zenoh.io) over TCP instead — see [Zenoh transport](#zenoh-transport) below.

---

## Install

**Python**

```bash
pip install .
```

Requires Python ≥ 3.10. Dependencies: `websockets`, `numpy`, `msgpack`, `eclipse-zenoh`, `tqdm`.

**C++**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Requires CMake ≥ 3.18 and a C++14 compiler.
[ixwebsocket](https://github.com/machinezone/IXWebSocket),
[msgpack-cxx](https://github.com/msgpack/msgpack-c), and
[Eigen 3.4](https://eigen.tuxfamily.org) are fetched automatically via `FetchContent`. To consume from another CMake project:

```cmake
add_subdirectory(path/to/chiral)
target_link_libraries(my_target chiral)
```

---

## Usage — Python

### Server (robot side)

Subclass `PolicyServer` and implement `camera_configs`, `reset`, and `apply_action`. The base class pre-allocates one buffer and one `threading.Lock` per camera for images, depths, intrinsics, and extrinsics. Sensor threads write via the `update_*` helpers; `_make_obs()` snapshots everything consistently under each camera's lock.

```python
import threading, time
import numpy as np
import chiral

class MyServer(chiral.PolicyServer):
    def camera_configs(self) -> list[chiral.CameraConfig]:
        return [chiral.CameraConfig(
            name="wrist_cam",
            height=H, width=W, channels=3,
            has_depth=True,
            intrinsics=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),
            extrinsics=np.eye(4),   # initial pose; updated each frame below
        )]

    def proprio_configs(self) -> list[chiral.ProprioConfig]:  # optional
        return [chiral.ProprioConfig(name="joint_pos", size=7),
                chiral.ProprioConfig(name="joint_vel", size=7)]

    def __init__(self):
        super().__init__(host="0.0.0.0", port=8765)  # or protocol="zenoh" for Zenoh on port 7447
        for name in self.images:
            threading.Thread(target=self._camera_loop, args=(name,), daemon=True).start()
        for name in self.proprios:
            threading.Thread(target=self._proprio_loop, args=(name,), daemon=True).start()

    def _camera_loop(self, name: str) -> None:
        t = 0.0
        while True:
            frame = ...               # (H, W, 3) uint8 from hardware
            depth = ...               # (H, W) float32 from hardware
            T     = fk_solver(...)    # (4, 4) float64 camera-to-world from FK
            self.update_image(name, frame)
            self.update_depth(name, depth)
            self.update_extrinsics(name, T)   # sent in every observation
            time.sleep(1/30)

    def _proprio_loop(self, name: str) -> None:
        while True:
            state = robot.read_joints()       # np.ndarray shape (DOF,)
            self.update_proprio(name, state)
            time.sleep(1/500)

    async def get_metadata(self) -> dict:
        return {"action_shape": [1, 7], "cameras": ["wrist_cam"]}  # optional

    async def reset(self) -> tuple[chiral.Observation, dict]:
        return self._make_obs(timestamp=0.0), {}

    # get_obs() is inherited — default snapshots _make_obs() under camera locks.
    # Override if you need to block until a fresh frame arrives.

    async def apply_action(self, action: np.ndarray) -> None:
        robot.send_joint_command(action)   # fire-and-forget; no obs returned

MyServer().run()
```

Per-camera buffers (all `{name: np.ndarray}`, protected by the same per-camera lock):

| Buffer | Shape | Thread-safe writer |
|---|---|---|
| `self.images` | `(H, W, C)` uint8 | `update_image(name, arr)` |
| `self.depths` | `(H, W)` float32 | `update_depth(name, arr)` |
| `self.intrinsics` | `(3, 3)` float64 | `update_intrinsics(name, arr)` |
| `self.extrinsics` | `(4, 4)` float64 | `update_extrinsics(name, arr)` |

`self.proprios` is `{name: np.ndarray}` (1-D float32); writer is `update_proprio(name, arr)`.
On the client: `obs["wrist_cam"].intrinsics`, `.extrinsics`, `.image`, `.depth` and `obs.proprios["joint_pos"]`.

For async contexts use `await server.serve()` instead of `.run()`.

### Client (policy / inference side)

`PolicyClient` streams observations and dispatches actions on independent background threads, so chunked policy predictions never stall waiting for camera data.

```python
import threading
import numpy as np
import chiral

def policy_loop(env, stop):
    while not stop.is_set():
        obs = env.latest_obs           # latest obs from the obs stream thread
        if obs is None:
            continue
        actions = policy(obs)          # (N, D) float32 — chunked predictions
        for a in actions:
            env.put_action(a)          # enqueue; dispatched at fixed Hz

with chiral.PolicyClient("ws://localhost:8765") as env:
    meta = env.get_metadata()          # {"action_shape": [1, 7], "cameras": [...]}
    obs, info = env.reset()

    env.start_obs_stream(hz=30)        # thread 1: keeps latest_obs fresh
    env.start_action_dispatch(hz=10)   # thread 3: sends queued actions at 10 Hz

    stop = threading.Event()
    t = threading.Thread(target=policy_loop, args=(env, stop))
    t.start()
    # ... run for desired duration, then:
    stop.set(); t.join()
```

Observations can be accessed from any camera by name:

```python
obs = env.latest_obs
cam   = obs["wrist_cam"]   # raises KeyError if missing
image = cam.image          # (H, W, 3) uint8
depth = cam.depth          # (H, W) float32, or None
K     = cam.intrinsics     # (3, 3) float64 — fresh every obs
T     = cam.extrinsics     # (4, 4) float64 — fresh every obs
```

`obs.cameras` is a plain list for iteration.

---

## Usage — C++

### Server

Pass a `std::vector<CameraConfig>` to the constructor. The base class pre-allocates buffers for images, depths, intrinsics, and extrinsics per camera, each protected by the same per-camera `std::mutex`. Sensor threads write via the `update_*` helpers; `make_obs()` snapshots everything consistently under each lock.

```cpp
#include <chiral/server.hpp>
#include <Eigen/Dense>
#include <atomic>
#include <cmath>
#include <thread>

class MyServer : public chiral::PolicyServer {
    std::atomic<bool> running_{true};
public:
    MyServer(const std::string& host, int port)
        : PolicyServer(
            // camera configs — intrinsics/extrinsics are the initial values
            {[&]{ chiral::CameraConfig c;
                  c.name = "wrist_cam"; c.height = H; c.width = W;
                  c.has_depth = true;
                  c.intrinsics << fx, 0, cx, 0, fy, cy, 0, 0, 1;  // Eigen::Matrix3d
                  c.extrinsics = Eigen::Matrix4d::Identity();       // updated each frame
                  return c; }()},
            // proprio configs (optional)
            {{"joint_pos", 7}, {"joint_vel", 7}},
            host, port)
    {
        for (std::size_t i = 0; i < configs_.size(); ++i)
            std::thread(&MyServer::camera_loop, this, i).detach();
        for (std::size_t i = 0; i < proprio_configs_.size(); ++i)
            std::thread(&MyServer::proprio_loop, this, i).detach();
    }
    ~MyServer() { running_ = false; }

    chiral::InfoMap get_metadata() override {          // optional
        return {{"action_N", "1"}, {"action_D", "7"}};
    }

    std::pair<chiral::Observation, chiral::InfoMap> reset() override {
        return {make_obs(0.0), {}};
    }

    chiral::StepResult step(const chiral::Action& action) override {
        chiral::StepResult r;
        r.obs        = make_obs(timestamp_);  // snapshots all buffers under their locks
        r.reward     = 0.f;
        r.terminated = false;
        r.truncated  = false;
        return r;
    }

private:
    void camera_loop(std::size_t idx) {
        double t = 0.0;
        while (running_) {
            // update_image(idx, frame.data(), frame.size());
            // update_depth(idx, depth.data(), depth.size());

            // Update extrinsics every frame for moving cameras (e.g. wrist).
            Eigen::Matrix4d T = fk_solver.compute(joint_pos);  // camera-to-world
            update_extrinsics(idx, T);   // thread-safe; sent in every observation

            t += 1.0 / 30.0;
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }
    void proprio_loop(std::size_t idx) {
        while (running_) {
            Eigen::VectorXf state = robot.read_joints();
            update_proprio(idx, state.data(), state.size());
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
};

int main() { MyServer("0.0.0.0", 8765).run(); }
```

Per-camera buffers (all protected by the same per-camera lock):

| Buffer | Type | Thread-safe writer |
|---|---|---|
| `images_[i]` | `vector<uint8_t>` H×W×C | `update_image(i, ptr, len)` |
| `depths_[i]` | `vector<uint8_t>` H×W×float | `update_depth(i, ptr, len)` |
| `intrinsics_[i]` | `Eigen::Matrix3d` | `update_intrinsics(i, K)` |
| `extrinsics_[i]` | `Eigen::Matrix4d` | `update_extrinsics(i, T)` |

`proprios_[i]` (`vector<float>`) is updated via `update_proprio(i, ptr, count)`.
On the client: `obs["wrist_cam"].intrinsics`, `.extrinsics`, `.image` and `obs.proprio("joint_pos").data`.

### Client

> **Note:** The C++ client still uses the legacy coupled `step()` API and has not yet been updated to the decoupled streaming design.

```cpp
#include <chiral/client.hpp>

chiral::PolicyClient env("ws://localhost:8765");
env.connect();

auto meta      = env.get_metadata();    // InfoMap
auto reset_res = env.reset();           // {Observation, InfoMap}
chiral::Observation obs = reset_res.first;

while (true) {
    const auto& cam = obs["wrist_cam"]; // throws std::out_of_range if missing
    // cam.image, cam.image_shape
    // cam.intrinsics  — Eigen::Matrix3d (camera matrix K)
    // cam.extrinsics  — Eigen::Matrix4d (camera-to-world T)
    // cam.has_depth, cam.depth_data, cam.depth_shape
    // obs.proprio("joint_pos").data  — std::vector<float>

    chiral::Action action;
    action.N = 1; action.D = 7;
    action.data.assign(7, 0.0f);

    auto res = env.step(action);        // StepResult {obs, reward, terminated, truncated, info}
    obs = std::move(res.obs);
    if (res.terminated || res.truncated) break;
}

env.close();
```

---

## Protocol

Every frame:

```
[4 bytes LE: header_len] [header_len bytes: msgpack] [raw payload bytes]
```

| Direction        | Type                | Payload                                                                  |
|------------------|---------------------|--------------------------------------------------------------------------|
| Client → Server  | `metadata`          | _(empty)_                                                                |
| Client → Server  | `reset`             | _(empty)_                                                                |
| Client → Server  | `obs_request`       | _(empty)_                                                                |
| Client → Server  | `apply_action`      | float32 buffer `[D]` — fire-and-forget, no server response               |
| Server → Client  | `metadata_response` | _(empty)_; header carries `data` dict                                    |
| Server → Client  | `reset_response`    | images + depths + proprios; header has camera/proprio metadata + `info`  |
| Server → Client  | `obs_response`      | images + depths + proprios; header has camera/proprio metadata           |

Camera metadata (name, intrinsics, extrinsics, shape, dtype, byte offset/size) and proprio metadata (name, dtype, byte offset/size) live in the msgpack header; all raw buffers are appended to the payload in order.

---

## Examples

```bash
# Python — WebSocket (default)
uv run examples/python/server_example.py &
uv run examples/python/client_example.py

# Python — Zenoh
uv run examples/python/zenoh_server_example.py &
uv run examples/python/zenoh_client_example.py

# C++
cmake -S examples/cpp -B build_ex -DCMAKE_BUILD_TYPE=Release
cmake --build build_ex
./build_ex/server_example &
./build_ex/client_example
```

---

## Zenoh transport

Pass `protocol="zenoh"` to switch from WebSocket to Zenoh over TCP. The public interface is identical.

**Server:**
```python
class MyServer(chiral.PolicyServer):
    def __init__(self):
        super().__init__(host="0.0.0.0", port=7447, protocol="zenoh")
    ...

MyServer().run()
```

**Client:**
```python
with chiral.PolicyClient("tcp/localhost:7447", protocol="zenoh") as env:
    obs, info = env.reset()
    ...
```

Zenoh uses its own efficient binary framing over TCP, avoiding WebSocket's HTTP handshake and per-message masking overhead. The wire protocol (msgpack header + raw arrays) is unchanged.
