#pragma once
#include "types.hpp"
#include <memory>
#include <string>
#include <utility>

namespace chiral {

/**
 * WebSocket client — policy/inference side, gym-like interface.
 *
 *   PolicyClient env("ws://localhost:8765");
 *   env.connect();
 *   auto reset_res = env.reset();           // {obs, info}
 *   auto step_res  = env.step(action);      // {obs, reward, terminated, truncated, info}
 *   env.close();
 */
class PolicyClient {
public:
    explicit PolicyClient(std::string uri = "ws://localhost:8765");
    ~PolicyClient();

    PolicyClient& connect();
    void          close();

    InfoMap                         get_metadata();
    std::pair<Observation, InfoMap> reset();
    StepResult                      step(const Action& action);

protected:
    std::string uri_;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace chiral
