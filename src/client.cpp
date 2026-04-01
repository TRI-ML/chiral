#include "chiral/client.hpp"
#include "chiral/protocol.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include <ixwebsocket/IXWebSocket.h>

namespace chiral {

// ─── Impl ─────────────────────────────────────────────────────────────────────

struct PolicyClient::Impl {
    ix::WebSocket ws;

    std::mutex                       mtx;
    std::condition_variable          cv;
    std::queue<std::vector<uint8_t>> frames;

    bool        has_last_obs{false};
    Observation last_obs;

    void push(std::vector<uint8_t> f) {
        { std::lock_guard<std::mutex> lk(mtx); frames.push(std::move(f)); }
        cv.notify_one();
    }

    std::vector<uint8_t> pop() {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this] { return !frames.empty(); });
        auto f = std::move(frames.front());
        frames.pop();
        return f;
    }
};

// ─── Constructor / Destructor ─────────────────────────────────────────────────

PolicyClient::PolicyClient(std::string uri)
    : uri_(std::move(uri)), impl_(std::make_unique<Impl>()) {}

PolicyClient::~PolicyClient() { close(); }

// ─── connect / close ──────────────────────────────────────────────────────────

PolicyClient& PolicyClient::connect() {
    impl_->ws.setUrl(uri_);
    impl_->ws.disablePerMessageDeflate();
    impl_->ws.setOnMessageCallback([this](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Message && msg->binary)
            impl_->push(std::vector<uint8_t>(msg->str.begin(), msg->str.end()));
    });
    impl_->ws.start();

    bool warned    = false;
    auto t_start   = std::chrono::steady_clock::now();
    while (impl_->ws.getReadyState() != ix::ReadyState::Open) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!warned &&
            std::chrono::steady_clock::now() - t_start > std::chrono::milliseconds(500))
        {
            std::fprintf(stderr, "chiral: waiting for server at %s ...\n", uri_.c_str());
            warned = true;
        }
    }
    return *this;
}

void PolicyClient::close() {
    impl_->ws.stop();
}

// ─── gym-like API ─────────────────────────────────────────────────────────────

InfoMap PolicyClient::get_metadata() {
    auto enc = encode_metadata_request();
    impl_->ws.sendBinary(std::string(enc.begin(), enc.end()));
    auto data = impl_->pop();
    return decode_metadata_response(data.data(), data.size());
}

std::pair<Observation, InfoMap> PolicyClient::reset() {
    auto enc = encode_reset();
    impl_->ws.sendBinary(std::string(enc.begin(), enc.end()));
    auto data = impl_->pop();
    auto result = decode_reset_response(data.data(), data.size());
    impl_->last_obs     = result.first;
    impl_->has_last_obs = true;
    return result;
}

StepResult PolicyClient::step(const Action& action) {
    Action act = action;
    if (act.obs_timestamps.empty() && impl_->has_last_obs) {
        for (const auto& cam : impl_->last_obs.cameras)
            act.obs_timestamps[cam.name] = cam.timestamp;
    }
    auto enc = encode_action(act);
    impl_->ws.sendBinary(std::string(enc.begin(), enc.end()));
    auto data = impl_->pop();
    auto result = decode_step_response(data.data(), data.size());
    impl_->last_obs = result.obs;
    return result;
}

}  // namespace chiral
