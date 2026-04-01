#pragma once
#include "types.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace chiral {

/**
 * Abstract WebSocket server — robot/environment side.
 *
 * Subclass and implement reset() and step(), then call run() (blocking).
 * The client drives the loop: it sends reset/action requests; the server
 * responds with observations and step results.
 *
 * Pass a vector of CameraConfig to the constructor. The base class
 * pre-allocates images_ and depths_ with one mutex per camera so that
 * per-camera sensor threads can write concurrently via update_image() /
 * update_depth(). make_obs() snapshots each buffer under its lock.
 */
class PolicyServer {
public:
    PolicyServer(std::vector<CameraConfig>  cam_configs,
                 std::vector<ProprioConfig> proprio_configs = {},
                 std::string host = "0.0.0.0",
                 int port = 8765);
    virtual ~PolicyServer() = default;

    /// Override to expose static metadata (action shape, camera names, etc.).
    virtual InfoMap get_metadata() { return {}; }

    virtual std::pair<Observation, InfoMap> reset()                 = 0;
    virtual StepResult                      step(const Action& act) = 0;

    void run();  ///< Blocking.

protected:
    std::string host_;
    int         port_;

    std::vector<CameraConfig>  configs_;
    std::vector<ProprioConfig> proprio_configs_;

    /// Pre-allocated image buffers (H*W*C bytes each), indexed as configs_.
    std::vector<std::vector<uint8_t>> images_;

    /// Monotonic timestamp (seconds) of the latest frame written via update_image().
    std::vector<double> image_timestamps_;

    /// Pre-allocated depth buffers (H*W*sizeof(float) bytes each), indexed as configs_.
    /// Only populated for cameras with has_depth=true.
    std::vector<std::vector<uint8_t>> depths_;

    /// Per-camera intrinsics and extrinsics — mutable each step (e.g. a wrist
    /// camera's extrinsics change as the arm moves). Initialized from CameraConfig.
    std::vector<Eigen::Matrix3d> intrinsics_;
    std::vector<Eigen::Matrix4d> extrinsics_;

    /// Pre-allocated proprio buffers (float32 elements), indexed as proprio_configs_.
    std::vector<std::vector<float>> proprios_;

    /// Thread-safe write of a new image frame into images_[idx].
    void update_image(std::size_t idx, const uint8_t* data, std::size_t len);

    /// Thread-safe write of a new depth frame into depths_[idx].
    void update_depth(std::size_t idx, const uint8_t* data, std::size_t len);

    /// Thread-safe update of the intrinsics matrix for camera idx.
    void update_intrinsics(std::size_t idx, const Eigen::Matrix3d& K);

    /// Thread-safe update of the camera-to-world extrinsics for camera idx.
    /// Call every step for cameras that move (e.g. wrist cameras).
    void update_extrinsics(std::size_t idx, const Eigen::Matrix4d& T);

    /// Thread-safe write of new proprio data into proprios_[idx].
    void update_proprio(std::size_t idx, const float* data, std::size_t count);

    /// Snapshot all buffers under their per-camera/proprio locks and return an Observation.
    Observation make_obs(double timestamp = 0.0);

private:
    /// One mutex per camera; protects the corresponding images_[i] / depths_[i].
    std::vector<std::unique_ptr<std::mutex>> cam_mutexes_;

    /// One mutex per proprio stream; protects proprios_[i].
    std::vector<std::unique_ptr<std::mutex>> proprio_mutexes_;
};

}  // namespace chiral
