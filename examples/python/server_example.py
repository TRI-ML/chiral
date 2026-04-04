"""Example robot server using the streaming (decoupled) API.

Observations are served on demand via get_obs(); actions arrive
independently via apply_action() dispatched at a fixed Hz by the client.

Run first, then start client_example.py:

    uv run examples/python/server_example.py &
    uv run examples/python/client_example.py
"""
import threading
import time

import numpy as np
import chiral

H, W     = 480, 640
CAMERAS  = ["cam_0", "cam_1", "cam_2", "cam_3", "cam_4", "cam_5", "cam_6", "cam_7"]
DOF      = 7  # joint-space dimensionality

# Intrinsics / extrinsics per camera (dummy values).
INTRINSICS = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
EXTRINSICS = [np.eye(4, dtype=np.float64)] * len(CAMERAS)


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

    def proprio_configs(self) -> list[chiral.ProprioConfig]:
        return [
            chiral.ProprioConfig(name="joint_pos", size=DOF),
            chiral.ProprioConfig(name="joint_vel", size=DOF),
        ]

    def _camera_loop(self, name: str) -> None:
        """Simulates a sensor driver running at ~30 Hz per camera."""
        t = 0.0
        while self._running:
            # Simulated RGB frame and depth map — replace with real hardware capture.
            new_image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            new_depth = np.random.uniform(0.5, 5.0, (H, W)).astype(np.float32)  # metres
            self.update_image(name, new_image)
            self.update_depth(name, new_depth)

            # Update extrinsics every frame for moving cameras (e.g. wrist).
            T = np.eye(4, dtype=np.float64)
            T[0, 3] = 0.1 * np.sin(t)  # simulated oscillating translation
            self.update_extrinsics(name, T)

            t += 1 / 30
            time.sleep(1 / 30)

    def _proprio_loop(self, name: str) -> None:
        """Simulates a proprioception driver running at ~500 Hz."""
        while self._running:
            # state = robot.read_joints()
            # self.update_proprio(name, state)
            time.sleep(1 / 500)

    async def get_metadata(self) -> dict:
        return {"cameras": CAMERAS, "action_shape": [1, DOF]}

    async def reset(self) -> tuple[chiral.Observation, dict]:
        self._step  = 0
        self._t_sum = 0.0
        return self._make_obs(), {}

    async def get_obs(self) -> chiral.Observation:
        """Return a snapshot of the current sensor state."""
        return self._make_obs()

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


if __name__ == "__main__":
    MyRobotServer().run()
