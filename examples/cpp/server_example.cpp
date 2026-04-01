#include <chiral/server.hpp>
#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

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

static std::vector<chiral::ProprioConfig> make_proprio_configs() {
    return {{"joint_pos", DOF}, {"joint_vel", DOF}};
}

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

    chiral::InfoMap get_metadata() override {
        return {{"cameras", "cam_0,cam_1,cam_2,cam_3"}, {"action_N", "1"}, {"action_D", "7"}};
    }

    std::pair<chiral::Observation, chiral::InfoMap> reset() override {
        step_ = 0; t_sum_ = 0.0;
        return {make_obs(), {}};
    }

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

private:
    void camera_loop(std::size_t idx) {
        // Simulates a sensor driver running at ~30 Hz.
        double t = 0.0;
        const auto& c = configs_[idx];
        std::vector<uint8_t> image(c.height * c.width * c.channels, 0);
        std::vector<float>   depth(c.height * c.width, 0.f);
        while (running_) {
            // Simulated RGB frame — replace with real hardware capture.
            std::fill(image.begin(), image.end(),
                      static_cast<uint8_t>(128 + 127 * std::sin(t)));
            update_image(idx, image.data(), image.size());

            // Simulated depth map (metres) — replace with real hardware capture.
            std::fill(depth.begin(), depth.end(),
                      1.0f + 0.5f * static_cast<float>(std::sin(t)));
            update_depth(idx, reinterpret_cast<const uint8_t*>(depth.data()),
                         depth.size() * sizeof(float));

            // Update extrinsics every frame for moving cameras (e.g. wrist).
            // In real code: T = fk_solver.compute(joint_positions)
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T(0, 3) = 0.1 * std::sin(t);  // simulated oscillating translation
            update_extrinsics(idx, T);

            t += 1.0 / 30.0;
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }

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
};

int main() {
    MyRobotServer("0.0.0.0", 8765).run();
}
