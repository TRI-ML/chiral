<img src="assets/header.png" alt="Chiral" style="width:100%;max-width:100%;" />

# Chiral

!!! warning "Alpha"
    This project is in early alpha. APIs may change without notice.

Compact interface for robot policy evaluation.

Sensor observations — RGB images, depth maps, camera intrinsics, camera extrinsics, and proprioception — are streamed from a robot **server** to a **policy client** in a separate process. Observations and actions flow on independent channels so chunked policy predictions never stall waiting for camera data. All communication is handled by the library; only application logic needs to be implemented.

---

## Architecture

```
  ┌─────────────────┐   obs stream (30 Hz)    ┌──────────────────┐
  │  PolicyServer   │ ──────────────────────► │  PolicyClient    │
  │  (robot side)   │                         │  (policy side)   │
  │                 │ ◄────────────────────── │                  │
  └─────────────────┘  action dispatch (10Hz) └──────────────────┘
```

The server runs on the robot side. The client connects, calls `reset()` once to start an episode, then runs three concurrent threads: one that continuously polls for the latest observation, one that runs policy inference and enqueues action chunks, and one that dispatches actions to the robot at a fixed Hz.

---

## Quick Start

=== "Python"

    **Server (robot side)**

    ```python
    import threading, time
    import numpy as np
    import chiral

    H, W = 480, 640

    class MyServer(chiral.PolicyServer):
        def camera_configs(self):
            return [chiral.CameraConfig(
                name="wrist_cam", height=H, width=W, channels=3,
                has_depth=True,
                intrinsics=np.array([[600,0,320],[0,600,240],[0,0,1]], dtype=np.float64),
                extrinsics=np.eye(4, dtype=np.float64),
            )]

        def proprio_configs(self):
            return [chiral.ProprioConfig(name="joint_pos", size=7)]

        def __init__(self):
            super().__init__(host="0.0.0.0", port=8765)
            for name in self.images:
                threading.Thread(target=self._camera_loop, args=(name,), daemon=True).start()

        def _camera_loop(self, name: str):
            while True:
                frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
                depth = np.random.rand(H, W).astype(np.float32)
                self.update_image(name, frame)
                self.update_depth(name, depth)
                time.sleep(1 / 30)

        async def reset(self):
            return self._make_obs(), {}

        async def apply_action(self, action):
            pass  # send action to robot hardware

    MyServer().run()
    ```

    **Client (policy side)**

    ```python
    import threading
    import numpy as np
    import chiral

    def policy_loop(env, stop):
        while not stop.is_set():
            obs = env.latest_obs
            if obs is None:
                continue
            print(obs["wrist_cam"].image.shape, obs["wrist_cam"].intrinsics)
            actions = np.zeros([8, 7], dtype=np.float32)  # chunked predictions
            for a in actions:
                env.put_action(a)

    with chiral.PolicyClient("ws://localhost:8765") as env:
        obs, info = env.reset()
        env.start_obs_stream(hz=30)
        env.start_action_dispatch(hz=10)

        stop = threading.Event()
        t = threading.Thread(target=policy_loop, args=(env, stop))
        t.start()
        # ... run for desired duration, then:
        stop.set(); t.join()
    ```

=== "C++"

    **Server (robot side)**

    ```cpp
    #include <chiral/server.hpp>
    #include <Eigen/Dense>
    #include <atomic>
    #include <thread>

    class MyServer : public chiral::PolicyServer {
        std::atomic<bool> running_{true};
    public:
        MyServer() : PolicyServer(
            []{
                chiral::CameraConfig c;
                c.name = "wrist_cam"; c.height = 480; c.width = 640;
                c.channels = 3; c.has_depth = true;
                c.intrinsics << 600, 0, 320, 0, 600, 240, 0, 0, 1;
                c.extrinsics = Eigen::Matrix4d::Identity();
                return std::vector<chiral::CameraConfig>{c};
            }(),
            {{"joint_pos", 7}},
            "0.0.0.0", 8765)
        {
            for (std::size_t i = 0; i < configs_.size(); ++i)
                std::thread(&MyServer::camera_loop, this, i).detach();
        }
        ~MyServer() { running_ = false; }

        std::pair<chiral::Observation, chiral::InfoMap> reset() override {
            return {make_obs(0.0), {}};
        }

        // C++ still uses the legacy coupled step() API
        chiral::StepResult step(const chiral::Action&) override {
            chiral::StepResult r;
            r.obs = make_obs(); r.reward = 0.f;
            r.terminated = false; r.truncated = false;
            return r;
        }
    private:
        void camera_loop(std::size_t idx) {
            while (running_) {
                Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                update_extrinsics(idx, T);
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
            }
        }
    };

    int main() { MyServer().run(); }
    ```

    **Client (policy side)**

    ```cpp
    #include <chiral/client.hpp>
    #include <cstdio>

    int main() {
        chiral::PolicyClient env("ws://localhost:8765");
        env.connect();

        auto [obs, info] = env.reset();

        for (int i = 0; i < 100; ++i) {
            chiral::Action action;
            action.N = 1; action.D = 7;
            action.data.assign(7, 0.f);

            auto res = env.step(action);
            obs = std::move(res.obs);
            std::printf("step %d  reward=%.2f\n", i, res.reward);
            if (res.terminated || res.truncated) break;
        }

        env.close();
    }
    ```

---

## Next Steps

- [Installation](installation.md)
- [Server Guide](guide/server.md)
- [Client Guide](guide/client.md)
- [Type Reference](reference/types.md)
- [Protocol Reference](reference/protocol.md)
- [Examples](examples/python.md)
