# Server Guide

The server runs on the robot or simulator side; it owns the sensor hardware and the environment state. The client drives the loop by sending `reset` and `step` requests; the server responds with observations.

---

## Threading Model

The server runs **one network thread** (the asyncio event loop) that handles WebSocket messages. When the client sends a `step` action, the network thread calls your `step()` implementation, which calls `make_obs()` / `_make_obs()` to snapshot the current sensor state.

Meanwhile, **one or more sensor threads** run independently: a camera capture thread reads frames from the hardware at 30 Hz; a proprioception thread reads joint state at 500 Hz. These threads write into the pre-allocated buffers using the `update_*` helpers.

The key invariant is that a sensor thread and the network thread must never corrupt a buffer by writing and reading it simultaneously. Chiral solves this with **one mutex per camera** that protects all four buffers for that camera (image pixels, depth map, intrinsics, extrinsics), and **one mutex per proprio stream**.

```
                ┌─────────────────────────────────┐
  camera thread │  update_image / update_depth     │
  (30 Hz)       │  update_intrinsics / update_extr │──► images_[i], depths_[i]
                │  (acquire cam_mutex_[i])          │    intrinsics_[i], extrinsics_[i]
                └────────────────────┬────────────┘
                                     │ same mutex
                ┌────────────────────▼────────────┐
  network thread│  make_obs() / _make_obs()        │
  (on step)     │  (acquire cam_mutex_[i])         │──► Observation snapshot
                └─────────────────────────────────┘
```

Because each camera has its own independent mutex, multiple camera threads can write concurrently without blocking each other. The snapshot in `make_obs()` acquires each camera's lock in turn, copies the buffers, and releases the lock before moving to the next camera.

!!! warning "Do not hold a camera lock across slow operations"
    The `update_*` helpers acquire the lock, copy the data, and release immediately. Never hold the lock across a hardware read (e.g. blocking on a USB transfer) or you will stall the network thread.

---

## Step 1 — Declare Cameras

The first thing to implement is `camera_configs()` (Python) or pass a `vector<CameraConfig>` to the constructor (C++). This is called once at construction time. The base class uses the returned configs to pre-allocate all buffers and mutexes before any step is taken.

=== "Python"

    ```python
    import numpy as np
    import chiral

    class MyServer(chiral.PolicyServer):

        def camera_configs(self) -> list[chiral.CameraConfig]:
            # Intrinsics: standard pinhole camera matrix K
            #   [fx  0  cx]
            #   [ 0 fy  cy]
            #   [ 0  0   1]
            K = np.array([
                [600.0,   0.0, 320.0],
                [  0.0, 600.0, 240.0],
                [  0.0,   0.0,   1.0],
            ], dtype=np.float64)

            return [
                chiral.CameraConfig(
                    name="wrist_cam",   # unique string identifier
                    height=480,         # image height in pixels
                    width=640,          # image width in pixels
                    channels=3,         # 3 for RGB, 1 for grayscale
                    has_depth=True,     # set True to allocate a depth buffer too
                    intrinsics=K,       # (3,3) float64 — initial K; updated each frame if zoom changes
                    extrinsics=np.eye(4, dtype=np.float64),  # (4,4) float64 — initial camera-to-world T
                    image_dtype=np.uint8,     # pixel dtype; uint8 is the most common choice
                    depth_dtype=np.float32,   # depth dtype; float32 in metres
                ),
                chiral.CameraConfig(
                    name="head_cam",
                    height=720, width=1280, channels=3,
                    has_depth=False,   # no depth for this camera — no depth buffer allocated
                    intrinsics=np.eye(3, dtype=np.float64),
                    extrinsics=np.eye(4, dtype=np.float64),
                ),
            ]
    ```

=== "C++"

    ```cpp
    #include <chiral/server.hpp>
    #include <Eigen/Dense>

    static std::vector<chiral::CameraConfig> make_cam_configs() {
        // Intrinsics: standard pinhole camera matrix K
        Eigen::Matrix3d K;
        K << 600.0,   0.0, 320.0,
               0.0, 600.0, 240.0,
               0.0,   0.0,   1.0;

        chiral::CameraConfig wrist;
        wrist.name      = "wrist_cam";  // unique string identifier
        wrist.height    = 480;          // image height in pixels
        wrist.width     = 640;          // image width in pixels
        wrist.channels  = 3;            // 3 for RGB, 1 for grayscale
        wrist.has_depth = true;         // allocate a depth buffer too
        wrist.intrinsics = K;           // Eigen::Matrix3d — initial K
        wrist.extrinsics = Eigen::Matrix4d::Identity();  // initial camera-to-world T

        chiral::CameraConfig head;
        head.name      = "head_cam";
        head.height    = 720;
        head.width     = 1280;
        head.channels  = 3;
        head.has_depth = false;  // no depth buffer for this camera
        head.intrinsics = Eigen::Matrix3d::Identity();
        head.extrinsics = Eigen::Matrix4d::Identity();

        return {wrist, head};
    }
    ```

### CameraConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` / `string` | — | Unique camera identifier. Used as the key in `obs["name"]`. |
| `height` | `int` | — | Image height in pixels. |
| `width` | `int` | — | Image width in pixels. |
| `channels` | `int` | 3 | Number of color channels (3 for RGB, 1 for grayscale). |
| `has_depth` | `bool` | `False` | Whether to allocate and stream a depth buffer. |
| `intrinsics` | `ndarray(3,3)` / `Matrix3d` | identity | Initial camera matrix K. Updated per-frame if the lens zooms. |
| `extrinsics` | `ndarray(4,4)` / `Matrix4d` | identity | Initial camera-to-world transform T. Updated per-frame for moving cameras. |
| `image_dtype` | `np.dtype` | `np.uint8` | Pixel element dtype. (Python only; C++ always uses `uint8_t`.) |
| `depth_dtype` | `np.dtype` | `np.float32` | Depth element dtype. (Python only; C++ always uses `float` / `float32`.) |

!!! note "Intrinsics and extrinsics are initial values"
    The values you pass in `CameraConfig` are used to initialize the internal buffers. They are **not** locked in — call `update_intrinsics` or `update_extrinsics` from your sensor thread each frame to keep them current. Both are included in every observation sent to the client.

---

## Step 2 — Declare Proprioception (Optional)

If your robot reads joint positions, velocities, torques, or any other 1-D float32 stream, add `ProprioConfig` entries. The base class pre-allocates a separate buffer and a separate mutex for each stream.

=== "Python"

    ```python
    def proprio_configs(self) -> list[chiral.ProprioConfig]:
        return [
            chiral.ProprioConfig(name="joint_pos", size=7),  # 7-DOF arm positions (rad)
            chiral.ProprioConfig(name="joint_vel", size=7),  # 7-DOF arm velocities (rad/s)
            chiral.ProprioConfig(name="ee_pose",  size=7),   # end-effector pose (x,y,z,qx,qy,qz,qw)
        ]
    ```

=== "C++"

    ```cpp
    // Pass directly to the PolicyServer constructor as the second argument
    static std::vector<chiral::ProprioConfig> make_proprio_configs() {
        return {
            {"joint_pos", 7},  // name, size
            {"joint_vel", 7},
            {"ee_pose",   7},
        };
    }
    ```

### ProprioConfig Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` / `string` | Unique stream identifier. Used as the key in `obs.proprios["name"]` (Python) or `obs.proprio("name")` (C++). |
| `size` | `int` | Number of float32 elements in the vector. |
| `dtype` | `np.dtype` | Element dtype. Default `np.float32`. (Python only.) |

If `proprio_configs()` returns an empty list (the default), no proprio buffers are allocated and `obs.proprios` will be empty on the client.

---

## Step 3 — Start Sensor Threads

After calling `super().__init__()` (Python) or after the base class constructor runs (C++), all buffers and locks are ready. This is the right moment to launch sensor threads.

=== "Python"

    ```python
    import threading

    class MyServer(chiral.PolicyServer):

        def camera_configs(self):
            ...  # (as above)

        def proprio_configs(self):
            ...  # (as above)

        def __init__(self):
            # Must call super().__init__() first — it calls camera_configs() and
            # proprio_configs() and pre-allocates self.images, self.depths,
            # self.intrinsics, self.extrinsics, self.proprios, and all locks.
            super().__init__(host="0.0.0.0", port=8765)

            self._running = True

            # Launch one capture thread per camera.
            # self.images is a dict keyed by camera name, so iterating its keys
            # gives exactly the names declared in camera_configs().
            for name in self.images:
                threading.Thread(
                    target=self._camera_loop,
                    args=(name,),
                    daemon=True,   # thread dies when the main process exits
                ).start()

            # Launch one update thread per proprio stream.
            for name in self.proprios:
                threading.Thread(
                    target=self._proprio_loop,
                    args=(name,),
                    daemon=True,
                ).start()
    ```

=== "C++"

    ```cpp
    #include <atomic>
    #include <thread>

    class MyServer : public chiral::PolicyServer {
        std::atomic<bool> running_{true};

    public:
        MyServer(const std::string& host, int port)
            : PolicyServer(make_cam_configs(), make_proprio_configs(), host, port)
        {
            // configs_ and proprio_configs_ are populated by the base class
            // constructor, so we can iterate them here safely.

            // Launch one capture thread per camera (indexed by position).
            for (std::size_t i = 0; i < configs_.size(); ++i)
                std::thread(&MyServer::camera_loop, this, i).detach();

            // Launch one update thread per proprio stream.
            for (std::size_t i = 0; i < proprio_configs_.size(); ++i)
                std::thread(&MyServer::proprio_loop, this, i).detach();
        }

        ~MyServer() { running_ = false; }
    };
    ```

!!! tip "Daemon threads in Python"
    Mark all sensor threads as `daemon=True` so they are automatically killed when the main process exits. If you need clean shutdown, set a `threading.Event` and check it in the loop.

---

## Step 4 — Sensor Threads: Fill Buffers

Each sensor thread runs a tight loop that reads data from hardware and calls the appropriate `update_*` helper. Every helper acquires the per-camera (or per-proprio) mutex, copies the data into the pre-allocated buffer, and releases the mutex — all in one atomic operation.

### Buffer Update Reference

| Helper | Updates | When to call |
|--------|---------|--------------|
| `update_image` | Image pixel buffer | Every camera frame (e.g. 30 Hz) |
| `update_depth` | Depth map buffer | Every camera frame for depth-capable cameras |
| `update_intrinsics` | 3×3 camera matrix K | When focal length or zoom changes |
| `update_extrinsics` | 4×4 camera-to-world T | Every frame for any camera that moves (wrist, head, etc.) |
| `update_proprio` | Proprioception float vector | Every state read (e.g. 500 Hz for joint encoders) |

### update_image

Writes new pixel data into the camera's image buffer.

=== "Python"

    ```python
    def _camera_loop(self, name: str) -> None:
        while self._running:
            # Read a new frame from the camera hardware.
            frame = camera_driver.capture_rgb(name)   # np.ndarray (H, W, 3) uint8

            # update_image acquires the per-camera lock, copies 'frame' into
            # self.images[name] in-place, then releases the lock.
            self.update_image(name, frame)

            time.sleep(1 / 30)
    ```

=== "C++"

    ```cpp
    void camera_loop(std::size_t idx) {
        while (running_) {
            // Read a new frame from the camera hardware.
            // frame must contain exactly H*W*C bytes.
            std::vector<uint8_t> frame = camera_driver.capture_rgb(idx);

            // update_image acquires cam_mutexes_[idx], memcpy's frame into
            // images_[idx], then releases the mutex.
            update_image(idx, frame.data(), frame.size());

            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }
    ```

### update_depth

Writes new depth data. Only valid for cameras declared with `has_depth=True`.

=== "Python"

    ```python
    # Depth is float32, shape (H, W), values in metres.
    depth = camera_driver.capture_depth(name)   # np.ndarray (H, W) float32
    self.update_depth(name, depth)
    ```

=== "C++"

    ```cpp
    // depth_bytes must contain exactly H*W*sizeof(float) bytes.
    std::vector<uint8_t> depth_bytes = camera_driver.capture_depth(idx);
    update_depth(idx, depth_bytes.data(), depth_bytes.size());
    ```

### update_intrinsics

Updates the camera matrix K. Call this when the focal length changes (e.g. zoom lens, autofocus that shifts the principal point). For fixed lenses you can skip this entirely — the initial value from `CameraConfig` is used.

=== "Python"

    ```python
    # K is a (3, 3) float64 ndarray.
    K = lens_driver.get_current_intrinsics(name)   # np.ndarray (3, 3) float64
    self.update_intrinsics(name, K)
    ```

=== "C++"

    ```cpp
    Eigen::Matrix3d K = lens_driver.get_current_intrinsics(idx);
    update_intrinsics(idx, K);
    ```

### update_extrinsics

Updates the camera-to-world transform T. **Call this every frame for any camera that moves with the robot arm** (e.g. a wrist-mounted camera). The canonical pattern is to compute T from a forward kinematics (FK) solver using the current joint positions.

=== "Python"

    ```python
    def _camera_loop(self, name: str) -> None:
        while self._running:
            frame = camera_driver.capture_rgb(name)
            depth = camera_driver.capture_depth(name)

            # Compute camera-to-world transform from current joint state.
            # fk_solver.compute() returns a (4, 4) float64 ndarray.
            joint_pos = robot.read_joints()
            T = fk_solver.compute(joint_pos, camera_frame=name)

            self.update_image(name, frame)
            self.update_depth(name, depth)
            self.update_extrinsics(name, T)   # update last so K and T are consistent

            time.sleep(1 / 30)
    ```

=== "C++"

    ```cpp
    void camera_loop(std::size_t idx) {
        while (running_) {
            std::vector<uint8_t> frame = camera_driver.capture_rgb(idx);
            std::vector<uint8_t> depth = camera_driver.capture_depth(idx);

            // Forward kinematics: T is the camera-to-world homogeneous transform.
            Eigen::VectorXf joints = robot.read_joints();
            Eigen::Matrix4d T = fk_solver.compute(joints, configs_[idx].name);

            update_image(idx, frame.data(), frame.size());
            update_depth(idx, depth.data(), depth.size());
            update_extrinsics(idx, T);   // update last for consistency

            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }
    ```

!!! note "Extrinsics convention"
    Chiral uses the **camera-to-world** convention: `T` maps a point in camera coordinates to world coordinates. This is the same convention used by most robotics FK solvers and NeRF-style representations. If your FK solver returns world-to-camera, take the inverse before passing to `update_extrinsics`.

### update_proprio

Updates a proprioception buffer. The vector length must match the `size` declared in `ProprioConfig`.

=== "Python"

    ```python
    def _proprio_loop(self, name: str) -> None:
        while self._running:
            state = robot.read_joints()   # np.ndarray shape (DOF,) float32
            self.update_proprio(name, state)
            time.sleep(1 / 500)          # 500 Hz joint encoder loop
    ```

=== "C++"

    ```cpp
    void proprio_loop(std::size_t idx) {
        const int n = proprio_configs_[idx].size;
        while (running_) {
            Eigen::VectorXf state = robot.read_joints();  // length n
            update_proprio(idx, state.data(), state.size());
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
    ```

---

## Step 5 — Implement reset and step

With the sensor threads running, `reset()` and `step()` are straightforward. Both simply call `_make_obs(timestamp)` to take a consistent snapshot of all buffers, then return the result.

`_make_obs()` / `make_obs()` acquires each camera's mutex in declaration order, copies the four buffers (image, depth, intrinsics, extrinsics) into a new `CameraInfo`, releases the mutex, and moves on to the next camera. It then does the same for each proprio stream. The result is an `Observation` where all buffers represent a single coherent point in time.

=== "Python"

    ```python
    import time

    async def reset(self) -> tuple[chiral.Observation, dict]:
        # Reset any environment state (counters, randomisation, etc.)
        self._step_count = 0

        # Snapshot the current sensor state.
        obs = self._make_obs(timestamp=0.0)
        return obs, {}

    async def step(
        self, action: np.ndarray
    ) -> tuple[chiral.Observation, float, bool, bool, dict]:
        # action is a (N, D) float32 ndarray decoded from the client message.
        # Execute the action on the robot hardware.
        robot.apply_action(action)

        # Snapshot sensor state with the current wall-clock timestamp.
        timestamp = self._step_count * 0.05  # or time.time()
        obs = self._make_obs(timestamp=timestamp)

        reward = compute_reward(obs)
        self._step_count += 1
        terminated = self._step_count >= 200

        return obs, reward, terminated, False, {}
    ```

=== "C++"

    ```cpp
    std::pair<chiral::Observation, chiral::InfoMap> reset() override {
        step_ = 0;
        // Snapshot the current sensor state at timestamp 0.
        return {make_obs(0.0), {}};
    }

    chiral::StepResult step(const chiral::Action& action) override {
        // action.data is a vector<float> of length action.N * action.D
        robot.apply_action(action.data);

        chiral::StepResult r;
        r.obs        = make_obs(step_ * 0.05);   // snapshot all buffers
        r.reward     = compute_reward(r.obs);
        r.terminated = ++step_ >= 200;
        r.truncated  = false;
        r.info       = {};
        return r;
    }
    ```

!!! tip "Timestamp conventions"
    The `timestamp` field is passed through to the client as-is. Use `time.time()` / `std::chrono` for wall-clock time, or a step counter multiplied by your control dt for episode time. The client can access it as `obs.timestamp`.

---

## Step 6 — Start the Server

=== "Python"

    For blocking usage (the standard case), call `run()` from the main thread or from `__main__`:

    ```python
    if __name__ == "__main__":
        MyServer().run()
    ```

    For async contexts (e.g. if you already have an asyncio event loop), use `await serve()` instead:

    ```python
    async def main():
        server = MyServer()
        await server.serve()   # runs forever; integrate with your existing event loop

    asyncio.run(main())
    ```

=== "C++"

    `run()` is blocking. Call it from `main()`:

    ```cpp
    int main() {
        MyServer("0.0.0.0", 8765).run();
    }
    ```

    The server listens on `0.0.0.0` (all interfaces) by default. To restrict to localhost only, pass `"127.0.0.1"` as the host. The default port is `8765`; pass a different value to the constructor if needed.

!!! note "One client at a time"
    Chiral is designed for a one-to-one connection between a single server and a single policy client. There is no broadcast or multi-client fanout.

### Alternative: Zenoh transport

Pass `protocol="zenoh"` to use Zenoh over TCP instead of WebSocket. All other code — `camera_configs`, `reset`, `step`, the `update_*` helpers — stays unchanged.

=== "Python"

    ```python
    def __init__(self):
        super().__init__(host="0.0.0.0", port=7447, protocol="zenoh")
    ```

The default port for Zenoh is `7447`. The WebSocket default (`8765`) is unchanged when `protocol` is omitted.

---

## Exposing Metadata (Optional)

`get_metadata()` allows the server to expose static information before the episode starts: action shape, camera names, calibration constants, etc. The client calls it once after connecting.

=== "Python"

    ```python
    async def get_metadata(self) -> dict:
        return {
            "cameras":      [c.name for c in self._configs],
            "action_shape": [1, 7],
            "action_space": "joint_position",
            "control_hz":   20,
        }
    ```

=== "C++"

    ```cpp
    chiral::InfoMap get_metadata() override {
        // InfoMap = unordered_map<string, string>.
        // All values are strings; non-string data should be stringified.
        return {
            {"cameras",       "wrist_cam,head_cam"},
            {"action_N",      "1"},
            {"action_D",      "7"},
            {"action_space",  "joint_position"},
            {"control_hz",    "20"},
        };
    }
    ```

!!! note "InfoMap is string-only"
    The C++ `InfoMap` type is `unordered_map<string, string>`. Stringify any non-string metadata (numbers, lists) before inserting. The Python `dict` returned by `get_metadata()` can hold arbitrary msgpack-compatible values.

---

## Complete Minimal Server

=== "Python"

    ```python
    import threading, time
    import numpy as np
    import chiral

    H, W, DOF = 480, 640, 7

    class MinimalServer(chiral.PolicyServer):

        def camera_configs(self):
            return [chiral.CameraConfig(
                name="cam", height=H, width=W, channels=3,
                has_depth=True,
                intrinsics=np.array([[600,0,320],[0,600,240],[0,0,1]], dtype=np.float64),
                extrinsics=np.eye(4, dtype=np.float64),
            )]

        def proprio_configs(self):
            return [chiral.ProprioConfig(name="joint_pos", size=DOF)]

        def __init__(self):
            super().__init__()
            self._step = 0
            for name in self.images:
                threading.Thread(target=self._cam, args=(name,), daemon=True).start()
            for name in self.proprios:
                threading.Thread(target=self._prop, args=(name,), daemon=True).start()

        def _cam(self, name):
            while True:
                frame = np.zeros((H, W, 3), dtype=np.uint8)   # replace with hardware read
                depth = np.zeros((H, W),    dtype=np.float32) # replace with hardware read
                T     = np.eye(4, dtype=np.float64)            # replace with FK
                self.update_image(name, frame)
                self.update_depth(name, depth)
                self.update_extrinsics(name, T)
                time.sleep(1 / 30)

        def _prop(self, name):
            while True:
                q = np.zeros(DOF, dtype=np.float32)   # replace with robot.read_joints()
                self.update_proprio(name, q)
                time.sleep(1 / 500)

        async def reset(self):
            self._step = 0
            return self._make_obs(), {}

        async def step(self, action):
            obs = self._make_obs(timestamp=self._step * 0.05)
            self._step += 1
            return obs, 0.0, self._step >= 200, False, {}

    if __name__ == "__main__":
        MinimalServer().run()
    ```

=== "C++"

    ```cpp
    #include <chiral/server.hpp>
    #include <Eigen/Dense>
    #include <atomic>
    #include <thread>

    static constexpr int H = 480, W = 640, DOF = 7;

    class MinimalServer : public chiral::PolicyServer {
        int               step_{0};
        std::atomic<bool> running_{true};
    public:
        MinimalServer()
            : PolicyServer(
                []{
                    chiral::CameraConfig c;
                    c.name = "cam"; c.height = H; c.width = W;
                    c.channels = 3; c.has_depth = true;
                    c.intrinsics << 600, 0, 320, 0, 600, 240, 0, 0, 1;
                    c.extrinsics = Eigen::Matrix4d::Identity();
                    return std::vector<chiral::CameraConfig>{c};
                }(),
                {{"joint_pos", DOF}},
                "0.0.0.0", 8765)
        {
            for (std::size_t i = 0; i < configs_.size(); ++i)
                std::thread(&MinimalServer::cam_loop, this, i).detach();
            for (std::size_t i = 0; i < proprio_configs_.size(); ++i)
                std::thread(&MinimalServer::prop_loop, this, i).detach();
        }
        ~MinimalServer() { running_ = false; }

        std::pair<chiral::Observation, chiral::InfoMap> reset() override {
            step_ = 0;
            return {make_obs(0.0), {}};
        }

        chiral::StepResult step(const chiral::Action&) override {
            chiral::StepResult r;
            r.obs = make_obs(step_ * 0.05);
            r.reward = 0.f; r.truncated = false;
            r.terminated = ++step_ >= 200;
            return r;
        }

    private:
        void cam_loop(std::size_t idx) {
            while (running_) {
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity(); // replace with FK
                update_extrinsics(idx, T);
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
            }
        }
        void prop_loop(std::size_t idx) {
            Eigen::VectorXf q = Eigen::VectorXf::Zero(proprio_configs_[idx].size);
            while (running_) {
                update_proprio(idx, q.data(), q.size());
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    };

    int main() { MinimalServer().run(); }
    ```
