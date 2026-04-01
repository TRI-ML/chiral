# Python Examples

Two transport variants are available; both use the same server and client interface.

```bash
# WebSocket (default)
uv run examples/python/server_example.py &
uv run examples/python/client_example.py

# Zenoh
uv run examples/python/zenoh_server_example.py &
uv run examples/python/zenoh_client_example.py
```

---

## Server Example

**File:** `examples/python/server_example.py`

This script simulates an 8-camera robot server with depth maps, proprioception, and updating extrinsics (simulating wrist cameras that oscillate). The full annotated source follows.

### Imports and Constants

```python
"""Example environment server.

Run first, then start client_example.py:

    uv run examples/python/server_example.py &
    uv run examples/python/client_example.py
"""
import threading
import time

import numpy as np
import chiral

H, W     = 480, 640         # camera resolution
CAMERAS  = ["cam_0", "cam_1", "cam_2", "cam_3",
            "cam_4", "cam_5", "cam_6", "cam_7"]   # 8 cameras
DOF      = 7                # joint-space dimensionality
```

`H` and `W` define the frame resolution. `DOF=7` is the dimensionality of the robot arm (7 joints). `CAMERAS` is the list of camera names — using a list here makes it easy to loop over them in `camera_configs`.

### Initial Calibration

```python
# Intrinsics / extrinsics per camera (dummy values).
INTRINSICS = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
EXTRINSICS = [np.eye(4, dtype=np.float64)] * len(CAMERAS)
```

A single shared intrinsics matrix is used for all cameras (same focal length and principal point). In a real system each camera would have its own calibrated K. The extrinsics start as the 4×4 identity, meaning all cameras are initially at the world origin — the camera loop updates them each frame.

### Class Definition and Constructor

```python
class MyRobotServer(chiral.PolicyServer):
    def __init__(self):
        super().__init__(host="0.0.0.0", port=8765)
        # self.images, self.depths, self.proprios, and all locks are
        # pre-allocated by the base class.
        self._step    = 0
        self._t_sum   = 0.0
        self._running = True

        # One capture thread per camera.
        for name in self.images:
            threading.Thread(
                target=self._camera_loop, args=(name,), daemon=True
            ).start()

        # One update thread per proprio stream.
        for name in self.proprios:
            threading.Thread(
                target=self._proprio_loop, args=(name,), daemon=True
            ).start()
```

The `super().__init__()` call is the critical first step. It calls `camera_configs()` and `proprio_configs()`, then pre-allocates all buffers (`self.images`, `self.depths`, `self.intrinsics`, `self.extrinsics`, `self.proprios`) and per-camera / per-proprio mutexes. Only after `super().__init__()` returns is it safe to iterate `self.images` and `self.proprios` to launch threads.

Note that threads are `daemon=True`: they are automatically killed when the main process exits, so no explicit shutdown is needed.

!!! warning "Call super().__init__() before accessing self.images"
    `self.images` and `self.proprios` are not defined until `super().__init__()` returns. Attempting to access them before calling `super()` will raise `AttributeError`.

### camera_configs

```python
    def camera_configs(self) -> list[chiral.CameraConfig]:
        return [
            chiral.CameraConfig(
                name=name,
                height=H,
                width=W,
                channels=3,
                has_depth=True,
                intrinsics=INTRINSICS,
                extrinsics=EXTRINSICS[i],
            )
            for i, name in enumerate(CAMERAS)
        ]
```

Returns one `CameraConfig` per camera. All cameras are `has_depth=True`, so the base class will allocate a `(H, W)` float32 depth buffer for each. The `intrinsics` and `extrinsics` fields here are the **initial** values; the camera loop updates `extrinsics` every frame.

### proprio_configs

```python
    def proprio_configs(self) -> list[chiral.ProprioConfig]:
        return [
            chiral.ProprioConfig(name="joint_pos", size=DOF),
            chiral.ProprioConfig(name="joint_vel", size=DOF),
        ]
```

Two proprio streams, each of length 7 (one value per joint). The base class pre-allocates `self.proprios["joint_pos"]` and `self.proprios["joint_vel"]` as zero `float32` arrays of length 7.

### Camera Loop

```python
    def _camera_loop(self, name: str) -> None:
        """Simulates a sensor driver running at ~30 Hz per camera."""
        t = 0.0
        while self._running:
            # ── hardware capture goes here ─────────────────────────────────
            # new_image = camera_driver.capture_image(name)   # (H, W, 3) uint8
            # new_depth = camera_driver.capture_depth(name)   # (H, W)    float32
            # self.update_image(name, new_image)
            # self.update_depth(name, new_depth)
            # ──────────────────────────────────────────────────────────────

            # Update extrinsics every frame for moving cameras (e.g. wrist).
            # In real code: T = fk_solver.compute(joint_positions)
            T = np.eye(4, dtype=np.float64)
            T[0, 3] = 0.1 * np.sin(t)   # simulated oscillating translation
            self.update_extrinsics(name, T)

            t += 1 / 30
            time.sleep(1 / 30)
```

Each camera runs its own thread at 30 Hz. The camera hardware calls are commented out for the example — they are where you would call your actual camera driver SDK.

The extrinsics update is the important part: `T[0, 3] = 0.1 * sin(t)` simulates the camera translating sinusoidally along the x axis, as a wrist camera would as the arm moves. `update_extrinsics` acquires the per-camera mutex, copies the 4×4 matrix into the internal buffer, and releases the mutex — so this is safe to call from any thread.

### Proprio Loop

```python
    def _proprio_loop(self, name: str) -> None:
        """Simulates a proprioception driver running at ~500 Hz."""
        while self._running:
            # ── hardware read goes here ────────────────────────────────────
            # state = robot.read_joints()  # np.ndarray shape (DOF,)
            # self.update_proprio(name, state)
            # ──────────────────────────────────────────────────────────────
            time.sleep(1 / 500)
```

Joint encoders typically run at 500 Hz or faster. This loop simulates the hardware read rate without actually doing anything (the proprio buffer stays at zeros). In a real system, replace the commented lines with the actual hardware call and `update_proprio` invocation.

### get_metadata

```python
    async def get_metadata(self) -> dict:
        return {"cameras": CAMERAS, "action_shape": [1, DOF]}
```

Returns the list of camera names and the expected action shape. The client reads this once at startup to configure itself correctly. Override this method to expose any static information the policy client needs before the episode starts.

### reset

```python
    async def reset(self) -> tuple[chiral.Observation, dict]:
        self._step  = 0
        self._t_sum = 0.0
        return self._make_obs(), {}
```

Resets the step counter and timing accumulator, then snapshots the current sensor state with `_make_obs()`. The second return value is an info dict (empty here). In a real system, you would also command the robot to its home configuration before returning.

### step

```python
    async def step(self, action: np.ndarray) -> tuple[chiral.Observation, float, bool, bool, dict]:
        t0 = time.perf_counter()

        # _make_obs() snapshots all camera and proprio buffers under their locks.
        obs = self._make_obs(timestamp=self._step * 0.05)

        step_ms = (time.perf_counter() - t0) * 1e3
        self._step  += 1
        self._t_sum += step_ms

        if self._step % 10 == 0:
            print(f"step={self._step:4d}  "
                  f"server_step={step_ms:5.2f}ms  "
                  f"avg={self._t_sum / self._step:5.2f}ms")

        terminated = self._step >= 200
        return obs, 0.0, terminated, False, {}
```

`_make_obs(timestamp)` is the key call. It acquires each camera's mutex in turn, copies the four buffers (image, depth, intrinsics, extrinsics) into new arrays, and releases the mutex before moving to the next camera. The result is a consistent snapshot even though sensor threads may be writing concurrently.

The timestamp is set to `step * 0.05`, representing 50 ms per step (20 Hz control). The episode ends after 200 steps (`terminated = True`). In a real system you would also apply `action` to the robot here.

### Entrypoint

```python
if __name__ == "__main__":
    MyRobotServer().run()
```

`run()` starts the asyncio event loop and listens for WebSocket connections indefinitely. It is blocking — nothing after this line executes until you kill the process.

---

## Client Example

**File:** `examples/python/client_example.py`

The client connects to the server, calls `reset()`, and then steps through the episode. A `tqdm` progress bar displays a rolling latency summary (current, mean, p95) and fps updated every step. A full percentile breakdown is printed at the end.

### Imports

```python
"""Example policy client.

Start server_example.py first, then run this:

    uv run examples/python/server_example.py &
    uv run examples/python/client_example.py
"""
import time
from collections import deque

import numpy as np
from tqdm import tqdm
import chiral
```

### Connection and Metadata

```python
if __name__ == "__main__":
    with chiral.PolicyClient("ws://localhost:8765") as env:
        meta = env.get_metadata()
        cameras      = meta.get("cameras", [])
        action_shape = meta.get("action_shape", [1, 7])
        print(f"cameras: {cameras}  action_shape: {action_shape}\n")
```

`PolicyClient` is used as a context manager: `__enter__` calls `connect()` and `__exit__` calls `close()`. The `connect()` call polls until the server accepts the WebSocket handshake, so you can start this script immediately after the server — it will wait automatically.

`get_metadata()` returns the dict from the server's `get_metadata()` override. Here we extract the camera list and action shape to configure the loop.

### Reset

```python
        obs, info = env.reset()
        total_reward = 0.0
        step         = 0
        t_episode    = time.perf_counter()
```

`reset()` sends a `reset` message and blocks until the `reset_response` arrives. The response includes a full observation with all camera images, depths, intrinsics, extrinsics, and proprio data.

### Episode Loop

```python
        while True:
            # Replace with real policy inference.
            action = np.zeros(action_shape, dtype=np.float32)
```

The action is constructed as zeros here. In a real policy this would be the output of your neural network inference step — e.g. `action = model.forward(obs)`.

### Stepping and Latency Measurement

```python
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            latency_ms = (time.perf_counter() - t0) * 1e3

            step += 1
            latencies.append(latency_ms)
            window.append(latency_ms)
            fps = step / (time.perf_counter() - t_episode)

            w = np.array(window)
            pbar.set_postfix(
                lat=f"{latency_ms:.0f}ms",
                mean=f"{w.mean():.0f}ms",
                p95=f"{np.percentile(w, 95):.0f}ms",
                fps=f"{fps:.1f}",
            )
            pbar.update(1)
```

`env.step(action)` serializes the action as float32 bytes, sends it, and blocks until the `step_response` arrives. `latency_ms` is the full round-trip: serialize → send → server processes → receive → deserialize.

The `tqdm` bar is updated every step with the current latency, a rolling mean and p95 over the last 20 steps, and throughput. All latencies are also accumulated in `latencies` for the final summary.

### Episode Summary

```python
        arr = np.array(latencies)
        fps = step / (time.perf_counter() - t_episode)
        print(f"\n── episode summary ──────────────────────────────")
        print(f"steps={step}  avg_fps={fps:.1f}")
        print(f"mean={arr.mean():.1f}ms  median={np.median(arr):.1f}ms")
        print(f"min={arr.min():.1f}ms   max={arr.max():.1f}ms")
        print(f"p95={np.percentile(arr, 95):.1f}ms  p99={np.percentile(arr, 99):.1f}ms")
```

Printed once after the `with tqdm(...)` block exits. Reports mean, median, min, max, p95, and p99 latency across the full episode.

### Termination

The loop ends when the server sets `terminated=True` (after 200 steps in the example).

---

## Expected Output

Running both scripts should produce output similar to:

**Server terminal:**
```
step=   1  server_step= 0.12ms  avg= 0.12ms
step=   2  server_step= 0.11ms  avg= 0.12ms
...
step= 200  server_step= 0.11ms  avg= 0.11ms
```

**Client terminal:**
```
cameras: ['cam_0', ..., 'cam_7']  action_shape: [1, 7]

 200/? [00:38<00:00,  5.2step/s, lat=189ms, mean=191ms, p95=210ms, fps=5.2]

── episode summary ──────────────────────────────
steps=200  avg_fps=5.2
mean=191.3ms  median=190.1ms
min=178.4ms   max=241.2ms
p95=210.5ms  p99=228.7ms
```

!!! tip "Interpreting latency"
    The latency includes the full round-trip: client serializes the action, sends it, the server calls `_make_obs()` to snapshot 8 cameras (8 × 480 × 640 × 3 = ~7.4 MB of RGB + ~7.4 MB of depth = ~15 MB per step), serializes the response, and the client deserializes it. Over loopback this is typically 150–250 ms for this data volume.

---

---

## Zenoh Examples

`examples/python/zenoh_server_example.py` and `zenoh_client_example.py` are identical in structure to the WebSocket examples above. The only differences are:

- Server: `super().__init__(protocol="zenoh")` — listens on `tcp/0.0.0.0:7447`
- Client: `PolicyClient("tcp/localhost:7447", protocol="zenoh")`

Everything else — camera configs, sensor threads, reset/step logic, tqdm latency measurement — is unchanged. Use them to benchmark WebSocket vs Zenoh side-by-side with identical workloads.

---

## Adapting for a Real Robot

To use these examples with real hardware, make the following changes in `server_example.py`:

1. **`_camera_loop`**: Replace the commented `camera_driver.capture_image` / `capture_depth` calls with your actual camera SDK calls.
2. **`_camera_loop`**: Replace `T = np.eye(4)` / `T[0,3] = ...` with a call to your FK solver: `T = fk_solver.compute(joint_positions)`.
3. **`_proprio_loop`**: Replace `time.sleep(1/500)` with `state = robot.read_joints()` + `self.update_proprio(name, state)`.
4. **`step`**: Apply `action` to the robot before calling `_make_obs()`.
5. **`reset`**: Move the robot to its home configuration before returning the observation.

In `client_example.py`:

1. Replace `action = np.zeros(action_shape, ...)` with your actual policy inference call.
