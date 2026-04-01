#include "chiral/protocol.hpp"

#include <cstring>
#include <msgpack.hpp>
#include <sstream>
#include <stdexcept>

namespace chiral {
namespace {

// Convert any msgpack object to a string. For STR objects this is a plain
// copy; for all other types (arrays, maps, numbers, booleans) it uses the
// msgpack stream operator so that non-string metadata values (e.g.
// "action_shape": [1,7]) round-trip safely into the C++ InfoMap.
static std::string obj_to_str(const msgpack::object& obj) {
    if (obj.type == msgpack::type::STR)
        return obj.as<std::string>();
    std::ostringstream oss;
    oss << obj;
    return oss.str();
}

inline uint32_t read_u32_le(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

inline void write_u32_le(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(v & 0xff);
    buf.push_back((v >> 8) & 0xff);
    buf.push_back((v >> 16) & 0xff);
    buf.push_back((v >> 24) & 0xff);
}

std::vector<uint8_t> make_frame(const msgpack::sbuffer& hdr,
                                 const uint8_t* payload, std::size_t payload_size) {
    uint32_t hlen = static_cast<uint32_t>(hdr.size());
    std::vector<uint8_t> out;
    out.reserve(4 + hlen + payload_size);
    write_u32_le(out, hlen);
    out.insert(out.end(), hdr.data(), hdr.data() + hlen);
    if (payload && payload_size)
        out.insert(out.end(), payload, payload + payload_size);
    return out;
}

void pack_cameras(msgpack::packer<msgpack::sbuffer>& pk,
                   const std::vector<CameraInfo>& cameras,
                   std::vector<uint8_t>& payload) {
    pk.pack_array(cameras.size());
    std::size_t offset = 0;
    for (const auto& cam : cameras) {
        int n_keys = 8 + (cam.has_depth ? 4 : 0);
        pk.pack_map(n_keys);
        pk.pack(std::string("name"));         pk.pack(cam.name);
        pk.pack(std::string("timestamp"));    pk.pack(cam.timestamp);
        pk.pack(std::string("intrinsics"));   pk.pack_array(9);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) pk.pack(cam.intrinsics(r, c));
        pk.pack(std::string("extrinsics"));   pk.pack_array(16);
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) pk.pack(cam.extrinsics(r, c));
        pk.pack(std::string("image_shape"));  pk.pack_array(cam.image_shape.size());
        for (int v : cam.image_shape)         pk.pack(v);
        pk.pack(std::string("image_dtype"));  pk.pack(cam.image_dtype);
        pk.pack(std::string("image_offset")); pk.pack(offset);
        pk.pack(std::string("image_size"));   pk.pack(cam.image.size());
        payload.insert(payload.end(), cam.image.begin(), cam.image.end());
        offset += cam.image.size();
        if (cam.has_depth) {
            pk.pack(std::string("depth_shape"));  pk.pack_array(cam.depth_shape.size());
            for (int v : cam.depth_shape)         pk.pack(v);
            pk.pack(std::string("depth_dtype"));  pk.pack(cam.depth_dtype);
            pk.pack(std::string("depth_offset")); pk.pack(offset);
            pk.pack(std::string("depth_size"));   pk.pack(cam.depth_data.size());
            payload.insert(payload.end(), cam.depth_data.begin(), cam.depth_data.end());
            offset += cam.depth_data.size();
        }
    }
}

void pack_proprios(msgpack::packer<msgpack::sbuffer>& pk,
                    const std::vector<ProprioInfo>& proprios,
                    std::vector<uint8_t>& payload) {
    pk.pack_array(proprios.size());
    for (const auto& p : proprios) {
        std::size_t nbytes = p.data.size() * sizeof(float);
        pk.pack_map(4);
        pk.pack(std::string("name"));   pk.pack(p.name);
        pk.pack(std::string("dtype"));  pk.pack(std::string("float32"));
        pk.pack(std::string("offset")); pk.pack(payload.size());
        pk.pack(std::string("size"));   pk.pack(nbytes);
        const auto* raw = reinterpret_cast<const uint8_t*>(p.data.data());
        payload.insert(payload.end(), raw, raw + nbytes);
    }
}

// ── shared obs-frame encoder ──────────────────────────────────────────────────
// n_extra: number of additional map entries appended after the base 5 keys.
// extra_fn: callable(pk) that packs those entries.
template<typename F>
std::vector<uint8_t> encode_obs_frame(const std::string& type,
                                       const Observation& obs,
                                       int n_extra, F extra_fn) {
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);
    std::vector<uint8_t> payload;

    pk.pack_map(5 + n_extra);
    pk.pack(std::string("type"));      pk.pack(type);
    pk.pack(std::string("timestamp")); pk.pack(obs.timestamp);
    pk.pack(std::string("extra"));     pk.pack_map(0);
    pk.pack(std::string("cameras"));   pack_cameras(pk, obs.cameras, payload);
    pk.pack(std::string("proprios"));  pack_proprios(pk, obs.proprios, payload);
    extra_fn(pk);

    return make_frame(sbuf, payload.data(), payload.size());
}

// ── shared frame parser ───────────────────────────────────────────────────────
struct Frame {
    msgpack::object_handle                             oh;
    std::unordered_map<std::string, msgpack::object>  map;
    const uint8_t*                                     payload{nullptr};
};

Frame parse_frame(const uint8_t* data, std::size_t size) {
    if (size < 4) throw std::runtime_error("chiral: truncated frame");
    uint32_t hlen = read_u32_le(data);
    if (size < 4 + hlen) throw std::runtime_error("chiral: truncated header");

    Frame f;
    f.oh      = msgpack::unpack(reinterpret_cast<const char*>(data + 4), hlen);
    f.payload = data + 4 + hlen;
    const auto& root = f.oh.get();
    for (uint32_t i = 0; i < root.via.map.size; ++i)
        f.map[root.via.map.ptr[i].key.as<std::string>()] = root.via.map.ptr[i].val;
    return f;
}

Observation unpack_obs(const Frame& f) {
    Observation obs;
    obs.timestamp = f.map.at("timestamp").as<double>();

    const auto& cams_obj = f.map.at("cameras");
    for (uint32_t ci = 0; ci < cams_obj.via.array.size; ++ci) {
        std::unordered_map<std::string, msgpack::object> cm;
        const auto& co = cams_obj.via.array.ptr[ci];
        for (uint32_t i = 0; i < co.via.map.size; ++i)
            cm[co.via.map.ptr[i].key.as<std::string>()] = co.via.map.ptr[i].val;

        CameraInfo cam;
        cam.name      = cm.at("name").as<std::string>();
        cam.timestamp = cm.count("timestamp") ? cm.at("timestamp").as<double>() : 0.0;
        auto intr = cm.at("intrinsics").as<std::vector<double>>();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) cam.intrinsics(r, c) = intr[r*3 + c];
        auto extr = cm.at("extrinsics").as<std::vector<double>>();
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) cam.extrinsics(r, c) = extr[r*4 + c];
        cam.image_shape = cm.at("image_shape").as<std::vector<int>>();
        cam.image_dtype = cm.at("image_dtype").as<std::string>();
        auto img_off  = cm.at("image_offset").as<std::size_t>();
        auto img_size = cm.at("image_size").as<std::size_t>();
        cam.image.assign(f.payload + img_off, f.payload + img_off + img_size);
        if (cm.count("depth_shape")) {
            cam.has_depth   = true;
            cam.depth_shape = cm.at("depth_shape").as<std::vector<int>>();
            cam.depth_dtype = cm.at("depth_dtype").as<std::string>();
            auto d_off  = cm.at("depth_offset").as<std::size_t>();
            auto d_size = cm.at("depth_size").as<std::size_t>();
            cam.depth_data.assign(f.payload + d_off, f.payload + d_off + d_size);
        }
        obs.cameras.push_back(std::move(cam));
    }

    if (f.map.count("proprios")) {
        const auto& props_obj = f.map.at("proprios");
        for (uint32_t pi = 0; pi < props_obj.via.array.size; ++pi) {
            std::unordered_map<std::string, msgpack::object> pm;
            const auto& po = props_obj.via.array.ptr[pi];
            for (uint32_t i = 0; i < po.via.map.size; ++i)
                pm[po.via.map.ptr[i].key.as<std::string>()] = po.via.map.ptr[i].val;
            ProprioInfo p;
            p.name = pm.at("name").as<std::string>();
            auto p_off  = pm.at("offset").as<std::size_t>();
            auto p_size = pm.at("size").as<std::size_t>();
            p.data.resize(p_size / sizeof(float));
            std::memcpy(p.data.data(), f.payload + p_off, p_size);
            obs.proprios.push_back(std::move(p));
        }
    }
    return obs;
}

InfoMap unpack_info(const Frame& f) {
    InfoMap info;
    if (f.map.count("info")) {
        const auto& iobj = f.map.at("info");
        for (uint32_t i = 0; i < iobj.via.map.size; ++i)
            info[iobj.via.map.ptr[i].key.as<std::string>()] =
                obj_to_str(iobj.via.map.ptr[i].val);
    }
    return info;
}

void pack_info(msgpack::packer<msgpack::sbuffer>& pk, const InfoMap& info) {
    pk.pack(std::string("info")); pk.pack_map(info.size());
    for (const auto& kv : info) { pk.pack(kv.first); pk.pack(kv.second); }
}

}  // namespace

// ── peek_type ─────────────────────────────────────────────────────────────────

std::string peek_type(const uint8_t* data, std::size_t size) {
    return parse_frame(data, size).map.at("type").as<std::string>();
}

// ── metadata ──────────────────────────────────────────────────────────────────

std::vector<uint8_t> encode_metadata_request() {
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);
    pk.pack_map(1);
    pk.pack(std::string("type")); pk.pack(std::string("metadata"));
    return make_frame(sbuf, nullptr, 0);
}

std::vector<uint8_t> encode_metadata_response(const InfoMap& data) {
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);
    pk.pack_map(2);
    pk.pack(std::string("type")); pk.pack(std::string("metadata_response"));
    pk.pack(std::string("data")); pk.pack_map(data.size());
    for (const auto& kv : data) { pk.pack(kv.first); pk.pack(kv.second); }
    return make_frame(sbuf, nullptr, 0);
}

InfoMap decode_metadata_response(const uint8_t* data, std::size_t size) {
    auto f = parse_frame(data, size);
    InfoMap result;
    if (f.map.count("data")) {
        const auto& dobj = f.map.at("data");
        for (uint32_t i = 0; i < dobj.via.map.size; ++i)
            result[dobj.via.map.ptr[i].key.as<std::string>()] =
                obj_to_str(dobj.via.map.ptr[i].val);
    }
    return result;
}

// ── reset ─────────────────────────────────────────────────────────────────────

std::vector<uint8_t> encode_reset() {
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);
    pk.pack_map(1);
    pk.pack(std::string("type")); pk.pack(std::string("reset"));
    return make_frame(sbuf, nullptr, 0);
}

std::vector<uint8_t> encode_reset_response(const Observation& obs, const InfoMap& info) {
    return encode_obs_frame("reset_response", obs, 1,
        [&](msgpack::packer<msgpack::sbuffer>& pk) { pack_info(pk, info); });
}

std::pair<Observation, InfoMap> decode_reset_response(const uint8_t* data, std::size_t size) {
    auto f = parse_frame(data, size);
    return {unpack_obs(f), unpack_info(f)};
}

// ── step ──────────────────────────────────────────────────────────────────────

std::vector<uint8_t> encode_action(const Action& action) {
    msgpack::sbuffer sbuf;
    msgpack::packer<msgpack::sbuffer> pk(sbuf);
    pk.pack_map(4);
    pk.pack(std::string("type"));           pk.pack(std::string("action"));
    pk.pack(std::string("shape"));          pk.pack_array(2); pk.pack(action.N); pk.pack(action.D);
    pk.pack(std::string("dtype"));          pk.pack(std::string("float32"));
    pk.pack(std::string("obs_timestamps")); pk.pack_map(action.obs_timestamps.size());
    for (const auto& kv : action.obs_timestamps) { pk.pack(kv.first); pk.pack(kv.second); }
    const auto* raw = reinterpret_cast<const uint8_t*>(action.data.data());
    return make_frame(sbuf, raw, action.data.size() * sizeof(float));
}

Action decode_action(const uint8_t* data, std::size_t size) {
    auto f = parse_frame(data, size);
    auto shape = f.map.at("shape").as<std::vector<int>>();
    Action action;
    action.N = shape[0]; action.D = shape[1];
    action.data.resize(action.N * action.D);
    std::memcpy(action.data.data(), f.payload, action.data.size() * sizeof(float));
    if (f.map.count("obs_timestamps")) {
        const auto& ts_obj = f.map.at("obs_timestamps");
        for (uint32_t i = 0; i < ts_obj.via.map.size; ++i)
            action.obs_timestamps[ts_obj.via.map.ptr[i].key.as<std::string>()] =
                ts_obj.via.map.ptr[i].val.as<double>();
    }
    return action;
}

std::vector<uint8_t> encode_step_response(
    const Observation& obs, float reward,
    bool terminated, bool truncated, const InfoMap& info)
{
    return encode_obs_frame("step_response", obs, 4,
        [&](msgpack::packer<msgpack::sbuffer>& pk) {
            pk.pack(std::string("reward"));     pk.pack(reward);
            pk.pack(std::string("terminated")); pk.pack(terminated);
            pk.pack(std::string("truncated"));  pk.pack(truncated);
            pack_info(pk, info);
        });
}

StepResult decode_step_response(const uint8_t* data, std::size_t size) {
    auto f = parse_frame(data, size);
    StepResult r;
    r.obs        = unpack_obs(f);
    r.reward     = f.map.count("reward")     ? f.map.at("reward").as<float>()    : 0.f;
    r.terminated = f.map.count("terminated") ? f.map.at("terminated").as<bool>() : false;
    r.truncated  = f.map.count("truncated")  ? f.map.at("truncated").as<bool>()  : false;
    r.info       = unpack_info(f);
    return r;
}

}  // namespace chiral
