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

### get_obs

```python
    async def get_obs(self) -> chiral.Observation:
        """Return a snapshot of the current sensor state."""
        return self._make_obs()
```

Called by the client's obs stream thread at a fixed Hz. `_make_obs()` acquires each camera's mutex in turn, copies the four buffers (image, depth, intrinsics, extrinsics) into new arrays, and releases the mutex before moving to the next camera. The result is a consistent snapshot even though sensor threads may be writing concurrently.

The default base-class implementation already calls `_make_obs()`, so this override is explicit for clarity. Override to block until a fresh hardware frame has arrived if strict frame-freshness guarantees are needed.

### apply_action

```python
    async def apply_action(self, action: np.ndarray) -> None:
        """Receive one action slice from the client and apply it to the robot."""
        t0 = time.perf_counter()

        # ── hardware command goes here ─────────────────────────────────────
        # robot.send_joint_command(action)
        # ──────────────────────────────────────────────────────────────────

        step_ms = (time.perf_counter() - t0) * 1e3
        self._step  += 1
        self._t_sum += step_ms

        print(f"action={self._step:4d}  "
              f"apply={step_ms:5.2f}ms  "
              f"avg={self._t_sum / self._step:5.2f}ms",
              flush=True)
```

Called by the client's action dispatch thread at the configured Hz. No observation is returned — this is fire-and-forget. The client dispatches actions independently of obs polling so the two channels never block each other.

### Entrypoint

```python
if __name__ == "__main__":
    MyRobotServer().run()
```

`run()` starts the asyncio event loop and listens for WebSocket connections indefinitely. It is blocking — nothing after this line executes until you kill the process.

---

## Client Example

**File:** `examples/python/client_example.py`

The client connects to the server, calls `reset()`, starts an obs stream and action dispatch thread, then runs a policy loop that reads the latest observation and enqueues action chunks.

### Imports and Constants

```python
import time
import threading

import numpy as np
import chiral

ACTION_HZ    = 10    # Hz at which actions are dispatched to the robot
OBS_HZ       = 30   # Hz at which observations are fetched from the server
CHUNK_SIZE   = 10   # number of actions predicted per policy call
DOF          = 7
TOTAL_STEPS  = 200  # actions to dispatch before stopping
```

`CHUNK_SIZE` matches the prediction horizon of the policy — e.g. a diffusion policy might predict 10 future actions at once, all of which are enqueued and then dispatched at `ACTION_HZ`.

### Policy Loop

```python
def policy_loop(env: chiral.PolicyClient, stop: threading.Event) -> None:
    inference_count = 0
    while not stop.is_set():
        obs = env.latest_obs
        if obs is None:
            time.sleep(0.01)
            continue

        # Show that real observation data is arriving.
        cam_info = ", ".join(
            f"{c.name}@{c.timestamp:.3f}s img={c.image.shape}"
            for c in obs.cameras[:2]
        )
        print(f"[policy #{inference_count}] obs ts={obs.timestamp:.3f}  cameras: {cam_info}")

        # Replace with real model inference:
        actions = np.zeros((CHUNK_SIZE, DOF), dtype=np.float32)

        for a in actions:
            env.put_action(a)

        inference_count += 1
        time.sleep(CHUNK_SIZE / ACTION_HZ * 0.9)
```

`env.latest_obs` is updated by the obs stream thread without blocking the policy loop. Once a non-None observation is available, inference runs and all predicted actions are enqueued via `put_action`. The `time.sleep` paces the policy loop so the queue doesn't grow unboundedly — in practice this is bounded by GPU compute time.

### Connection, Reset, and Streaming Startup

```python
with chiral.PolicyClient("ws://localhost:8765") as env:
    obs, info = env.reset()

    env.start_obs_stream(hz=OBS_HZ)
    env.start_action_dispatch(hz=ACTION_HZ)
```

`reset()` is called once to start the episode. `start_obs_stream` launches a background coroutine on the client's asyncio loop that polls `obs_request` / `obs_response` at `OBS_HZ`. `start_action_dispatch` launches a background coroutine that dequeues one action per tick from `_action_queue` and sends it as `apply_action` (fire-and-forget).

### Running and Stopping

```python
    stop = threading.Event()
    policy_thread = threading.Thread(target=policy_loop, args=(env, stop), daemon=True)
    policy_thread.start()

    t_start = time.perf_counter()
    dispatched = 0
    while dispatched < TOTAL_STEPS:
        time.sleep(1 / ACTION_HZ)
        dispatched += 1

    stop.set()
    policy_thread.join()

    elapsed = time.perf_counter() - t_start
    print(f"\ndispatched {TOTAL_STEPS} actions in {elapsed:.1f}s  "
          f"({TOTAL_STEPS / elapsed:.1f} Hz)")
```

The main thread counts dispatched actions and stops after `TOTAL_STEPS`. `close()` (called by the context manager `__exit__`) automatically cancels the obs stream and action dispatch tasks.

---

## Expected Output

Running both scripts should produce output similar to:

**Server terminal:**
```
action=   1  apply= 0.01ms  avg= 0.01ms
action=   2  apply= 0.01ms  avg= 0.01ms
...
action= 200  apply= 0.01ms  avg= 0.01ms
```

**Client terminal:**
```
cameras: ['cam_0', ..., 'cam_7']  action_shape: [1, 7]

[policy #0] obs ts=0.000  cameras: cam_0@1.234s img=(480, 640, 3), cam_1@1.234s img=(480, 640, 3)
[policy #1] obs ts=0.000  cameras: cam_0@1.268s img=(480, 640, 3), cam_1@1.268s img=(480, 640, 3)
...
dispatched 200 actions in 20.0s  (10.0 Hz)
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
