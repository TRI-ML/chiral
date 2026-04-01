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

    auto meta = env.get_metadata();
    std::printf("metadata:\n");
    for (const auto& kv : meta)
        std::printf("  %s: %s\n", kv.first.c_str(), kv.second.c_str());
    std::printf("\n");

    auto [obs, info] = env.reset();

    float total_reward = 0.f;
    int   step         = 0;
    auto  t_episode    = Clock::now();

    while (true) {
        // Build action with Eigen, then hand off to the API.
        Eigen::Matrix<float, 1, 7> action_mat = Eigen::Matrix<float, 1, 7>::Zero();
        // policy_net.forward(obs) → action_mat   (replace with real inference)

        chiral::Action action;
        action.N = 1; action.D = 7;
        action.data.assign(action_mat.data(), action_mat.data() + action_mat.size());

        auto   t0      = Clock::now();
        auto   res     = env.step(action);
        double latency = Ms(Clock::now() - t0).count();

        obs           = std::move(res.obs);
        total_reward += res.reward;
        ++step;

        double fps = step / Sec(Clock::now() - t_episode).count();

        // Print a full summary on the first step, then just timing stats.
        if (step == 1) {
            std::printf("cameras:\n");
            for (const auto& cam : obs.cameras) {
                double fx = cam.intrinsics(0, 0), fy = cam.intrinsics(1, 1);
                double cx = cam.intrinsics(0, 2), cy = cam.intrinsics(1, 2);
                Eigen::Vector3d pos = cam.extrinsics.col(3).head<3>();

                float depth_mean = 0.f;
                if (cam.has_depth) {
                    const float* dp = reinterpret_cast<const float*>(cam.depth_data.data());
                    std::size_t  n  = cam.depth_data.size() / sizeof(float);
                    for (std::size_t k = 0; k < n; ++k) depth_mean += dp[k];
                    if (n) depth_mean /= static_cast<float>(n);
                }

                std::printf("  %s(%dx%d)  fx=%.0f fy=%.0f cx=%.0f cy=%.0f"
                            "  pos=[%.2f %.2f %.2f]%s\n",
                            cam.name.c_str(),
                            cam.image_shape[0], cam.image_shape[1],
                            fx, fy, cx, cy,
                            pos.x(), pos.y(), pos.z(),
                            cam.has_depth
                                ? (" depth_mean=" + std::to_string(depth_mean) + "m").c_str()
                                : "");
            }

            if (!obs.proprios.empty()) {
                std::printf("proprios:\n");
                for (const auto& p : obs.proprios) {
                    Eigen::Map<const Eigen::VectorXf> v(p.data.data(), p.data.size());
                    std::printf("  %s[%zu]  norm=%.4f\n",
                                p.name.c_str(), p.data.size(), v.norm());
                }
            }
            std::printf("\n");
        }

        std::printf("step=%4d  latency=%6.1fms  fps=%6.1f\n", step, latency, fps);

        if (res.terminated || res.truncated) {
            std::printf("\ndone — steps=%d  total_reward=%.2f  avg_fps=%.1f\n",
                        step, total_reward, fps);
            break;
        }
    }

    env.close();
}
