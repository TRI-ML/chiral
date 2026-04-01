#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace chiral {

using InfoMap = std::unordered_map<std::string, std::string>;

/// Static description of one proprioception stream.
/// Pass a vector of these to PolicyServer's constructor so the base class
/// can pre-allocate the proprio buffers before any step is taken.
struct ProprioConfig {
    std::string name;
    int size{0};  ///< number of float32 elements
};

/// Static description of one camera channel.
/// Pass a vector of these to PolicyServer's constructor so the base class
/// can pre-allocate the image/depth buffers before any step is taken.
struct CameraConfig {
    std::string     name;
    int             height{0};
    int             width{0};
    int             channels{3};
    bool            has_depth{false};
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();  ///< camera matrix K
    Eigen::Matrix4d extrinsics = Eigen::Matrix4d::Identity();  ///< camera-to-world T
};

struct CameraInfo {
    std::string     name;
    double          timestamp{0.0};       ///< monotonic time when this frame was captured
    Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Identity();  ///< camera matrix K
    Eigen::Matrix4d extrinsics = Eigen::Matrix4d::Identity();  ///< camera-to-world T

    std::vector<uint8_t> image;           ///< raw pixel bytes
    std::vector<int>     image_shape;     ///< [H, W, C]
    std::string          image_dtype;     ///< e.g. "uint8"

    bool                 has_depth{false};
    std::vector<uint8_t> depth_data;      ///< raw depth bytes
    std::vector<int>     depth_shape;     ///< [H, W]
    std::string          depth_dtype;     ///< e.g. "float32"
};

struct ProprioInfo {
    std::string        name;
    std::vector<float> data;  ///< float32 values
};

struct Observation {
    std::vector<CameraInfo>  cameras;
    std::vector<ProprioInfo> proprios;
    double timestamp{0.0};

    /// Look up a camera by name. Throws std::out_of_range if not found.
    CameraInfo& operator[](const std::string& name) {
        for (auto& c : cameras) if (c.name == name) return c;
        throw std::out_of_range("chiral: camera not found: " + name);
    }
    const CameraInfo& operator[](const std::string& name) const {
        for (const auto& c : cameras) if (c.name == name) return c;
        throw std::out_of_range("chiral: camera not found: " + name);
    }

    /// Look up a proprio stream by name. Throws std::out_of_range if not found.
    ProprioInfo& proprio(const std::string& name) {
        for (auto& p : proprios) if (p.name == name) return p;
        throw std::out_of_range("chiral: proprio not found: " + name);
    }
    const ProprioInfo& proprio(const std::string& name) const {
        for (const auto& p : proprios) if (p.name == name) return p;
        throw std::out_of_range("chiral: proprio not found: " + name);
    }
};

struct Action {
    std::vector<float> data;  ///< N*D floats, row-major
    int N{0}, D{0};
    std::unordered_map<std::string, double> obs_timestamps;  ///< camera name → capture time
};

struct StepResult {
    Observation obs;
    float       reward{0.f};
    bool        terminated{false};
    bool        truncated{false};
    InfoMap     info;
};

}  // namespace chiral
