#include "chiral/server.hpp"
#include "chiral/protocol.hpp"

#include <chrono>
#include <ixwebsocket/IXWebSocketServer.h>

namespace chiral {

PolicyServer::PolicyServer(std::vector<CameraConfig>  cam_configs,
                           std::vector<ProprioConfig> proprio_configs,
                           std::string host, int port)
    : host_(std::move(host)), port_(port),
      configs_(std::move(cam_configs)),
      proprio_configs_(std::move(proprio_configs))
{
    const std::size_t nc = configs_.size();
    images_.resize(nc);
    image_timestamps_.resize(nc, 0.0);
    depths_.resize(nc);
    intrinsics_.resize(nc);
    extrinsics_.resize(nc);
    cam_mutexes_.reserve(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        const auto& c = configs_[i];
        images_[i].assign(static_cast<std::size_t>(c.height) * c.width * c.channels, 0);
        if (c.has_depth)
            depths_[i].assign(static_cast<std::size_t>(c.height) * c.width * sizeof(float), 0);
        intrinsics_[i] = c.intrinsics;
        extrinsics_[i] = c.extrinsics;
        cam_mutexes_.push_back(std::make_unique<std::mutex>());
    }

    const std::size_t np = proprio_configs_.size();
    proprios_.resize(np);
    proprio_mutexes_.reserve(np);
    for (std::size_t i = 0; i < np; ++i) {
        proprios_[i].assign(proprio_configs_[i].size, 0.f);
        proprio_mutexes_.push_back(std::make_unique<std::mutex>());
    }
}

void PolicyServer::update_image(std::size_t idx, const uint8_t* data, std::size_t len) {
    std::lock_guard<std::mutex> lk(*cam_mutexes_[idx]);
    std::copy(data, data + len, images_[idx].begin());
    image_timestamps_[idx] = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void PolicyServer::update_depth(std::size_t idx, const uint8_t* data, std::size_t len) {
    std::lock_guard<std::mutex> lk(*cam_mutexes_[idx]);
    std::copy(data, data + len, depths_[idx].begin());
}

void PolicyServer::update_intrinsics(std::size_t idx, const Eigen::Matrix3d& K) {
    std::lock_guard<std::mutex> lk(*cam_mutexes_[idx]);
    intrinsics_[idx] = K;
}

void PolicyServer::update_extrinsics(std::size_t idx, const Eigen::Matrix4d& T) {
    std::lock_guard<std::mutex> lk(*cam_mutexes_[idx]);
    extrinsics_[idx] = T;
}

void PolicyServer::update_proprio(std::size_t idx, const float* data, std::size_t count) {
    std::lock_guard<std::mutex> lk(*proprio_mutexes_[idx]);
    std::copy(data, data + count, proprios_[idx].begin());
}

Observation PolicyServer::make_obs(double timestamp) {
    Observation obs;
    obs.timestamp = timestamp;
    for (std::size_t i = 0; i < configs_.size(); ++i) {
        const auto& c = configs_[i];
        CameraInfo cam;
        cam.name        = c.name;
        cam.image_shape = {c.height, c.width, c.channels};
        cam.image_dtype = "uint8";
        cam.has_depth   = c.has_depth;
        if (c.has_depth) {
            cam.depth_shape = {c.height, c.width};
            cam.depth_dtype = "float32";
        }
        {
            std::lock_guard<std::mutex> lk(*cam_mutexes_[i]);
            cam.timestamp  = image_timestamps_[i]; // snapshot
            cam.intrinsics = intrinsics_[i];       // snapshot copy
            cam.extrinsics = extrinsics_[i];       // snapshot copy
            cam.image      = images_[i];           // snapshot copy
            if (c.has_depth)
                cam.depth_data = depths_[i];       // snapshot copy
        }
        obs.cameras.push_back(std::move(cam));
    }

    for (std::size_t i = 0; i < proprio_configs_.size(); ++i) {
        ProprioInfo p;
        p.name = proprio_configs_[i].name;
        {
            std::lock_guard<std::mutex> lk(*proprio_mutexes_[i]);
            p.data = proprios_[i];  // snapshot copy
        }
        obs.proprios.push_back(std::move(p));
    }
    return obs;
}

void PolicyServer::run() {
    ix::WebSocketServer server(port_, host_);
    server.disablePerMessageDeflate();

    server.setOnClientMessageCallback(
        [this](std::shared_ptr<ix::ConnectionState> /*state*/,
               ix::WebSocket& ws,
               const ix::WebSocketMessagePtr& msg) {

            if (msg->type != ix::WebSocketMessageType::Message || !msg->binary)
                return;

            const auto* raw = reinterpret_cast<const uint8_t*>(msg->str.data());
            const std::size_t sz = msg->str.size();
            const std::string type = peek_type(raw, sz);

            if (type == "metadata") {
                auto enc = encode_metadata_response(get_metadata());
                ws.sendBinary(std::string(enc.begin(), enc.end()));

            } else if (type == "reset") {
                auto result = reset();
                auto enc = encode_reset_response(result.first, result.second);
                ws.sendBinary(std::string(enc.begin(), enc.end()));

            } else if (type == "action") {
                auto action = decode_action(raw, sz);
                auto result = step(action);
                auto enc = encode_step_response(
                    result.obs, result.reward,
                    result.terminated, result.truncated, result.info);
                ws.sendBinary(std::string(enc.begin(), enc.end()));
            }
        });

    server.listenAndStart();
    server.wait();
}

}  // namespace chiral
