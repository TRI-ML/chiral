#pragma once
#include "types.hpp"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace chiral {

/// Peek at the "type" field of a frame without full decoding.
std::string peek_type(const uint8_t* data, std::size_t size);

// ── metadata ──────────────────────────────────────────────────────────────────

std::vector<uint8_t> encode_metadata_request();
std::vector<uint8_t> encode_metadata_response(const InfoMap& data);
InfoMap              decode_metadata_response(const uint8_t* data, std::size_t size);

// ── reset ─────────────────────────────────────────────────────────────────────

std::vector<uint8_t>            encode_reset();
std::vector<uint8_t>            encode_reset_response(const Observation& obs, const InfoMap& info);
std::pair<Observation, InfoMap> decode_reset_response(const uint8_t* data, std::size_t size);

// ── step ──────────────────────────────────────────────────────────────────────

std::vector<uint8_t> encode_action(const Action& action);
Action               decode_action(const uint8_t* data, std::size_t size);

std::vector<uint8_t> encode_step_response(
    const Observation& obs, float reward,
    bool terminated, bool truncated, const InfoMap& info);

StepResult decode_step_response(const uint8_t* data, std::size_t size);

}  // namespace chiral
