"""
Binary protocol (no compression):
    [4 bytes LE: header_len] [header_len bytes: msgpack] [raw payload bytes]

Client → Server:
    metadata     : {"type": "metadata"}
    reset        : {"type": "reset"}
    obs_request  : {"type": "obs_request"}
    apply_action : {"type": "apply_action", "shape": [D], "dtype": str} + float32 payload
                   (fire-and-forget — server sends no response)

Server → Client:
    metadata_response : {"type": "metadata_response", "data": dict}
    reset_response    : obs fields + {"info": dict}
    obs_response      : obs fields

Obs fields: type, timestamp, extra, cameras[{name, intrinsics(9), extrinsics(16),
    image_shape, image_dtype, image_offset, image_size,
    depth_shape?, depth_dtype?, depth_offset?, depth_size?}],
    proprios[{name, dtype, offset, size}]
"""
import struct

import msgpack
import numpy as np

from .types import CameraInfo, Observation


# ── helpers ───────────────────────────────────────────────────────────────────

def _encode_obs_frame(msg_type: str, obs: Observation, extra_fields: dict) -> bytes:
    parts: list[bytes] = []
    offset = 0
    cameras = []

    for cam in obs.cameras:
        img_bytes = cam.image.tobytes()
        info: dict = {
            "name": cam.name,
            "timestamp": cam.timestamp,
            "intrinsics": cam.intrinsics.flatten().tolist(),
            "extrinsics": cam.extrinsics.flatten().tolist(),
            "image_shape": list(cam.image.shape),
            "image_dtype": cam.image.dtype.name,
            "image_offset": offset,
            "image_size": len(img_bytes),
        }
        parts.append(img_bytes)
        offset += len(img_bytes)

        if cam.depth is not None:
            depth_bytes = cam.depth.tobytes()
            info.update({
                "depth_shape": list(cam.depth.shape),
                "depth_dtype": cam.depth.dtype.name,
                "depth_offset": offset,
                "depth_size": len(depth_bytes),
            })
            parts.append(depth_bytes)
            offset += len(depth_bytes)

        cameras.append(info)

    proprios = []
    for name, arr in obs.proprios.items():
        arr = np.asarray(arr, dtype=np.float32)
        raw = arr.tobytes()
        proprios.append({
            "name": name,
            "dtype": arr.dtype.name,
            "offset": offset,
            "size": len(raw),
        })
        parts.append(raw)
        offset += len(raw)

    header = msgpack.packb({
        "type": msg_type,
        "timestamp": obs.timestamp,
        "cameras": cameras,
        "proprios": proprios,
        "extra": obs.extra,
        **extra_fields,
    })
    payload = b"".join(parts)
    return struct.pack("<I", len(header)) + header + payload


def _decode_obs_frame(data: bytes) -> tuple[Observation, dict]:
    (hlen,) = struct.unpack_from("<I", data, 0)
    header = msgpack.unpackb(data[4:4 + hlen], raw=False)
    payload = data[4 + hlen:]

    cameras = []
    for c in header["cameras"]:
        img = np.frombuffer(
            payload[c["image_offset"]: c["image_offset"] + c["image_size"]],
            dtype=c["image_dtype"],
        ).reshape(c["image_shape"]).copy()

        depth = None
        if "depth_shape" in c:
            depth = np.frombuffer(
                payload[c["depth_offset"]: c["depth_offset"] + c["depth_size"]],
                dtype=c["depth_dtype"],
            ).reshape(c["depth_shape"]).copy()

        cameras.append(CameraInfo(
            name=c["name"],
            intrinsics=np.array(c["intrinsics"], dtype=np.float64).reshape(3, 3),
            extrinsics=np.array(c["extrinsics"], dtype=np.float64).reshape(4, 4),
            image=img,
            depth=depth,
            timestamp=float(c.get("timestamp", 0.0)),
        ))

    proprios = {}
    for p in header.get("proprios", []):
        proprios[p["name"]] = np.frombuffer(
            payload[p["offset"]: p["offset"] + p["size"]],
            dtype=p["dtype"],
        ).copy()

    obs = Observation(
        cameras=cameras,
        proprios=proprios,
        timestamp=header["timestamp"],
        extra=header.get("extra", {}),
    )
    return obs, header


def peek_type(data: bytes) -> str:
    (hlen,) = struct.unpack_from("<I", data, 0)
    return msgpack.unpackb(data[4:4 + hlen], raw=False)["type"]


# ── metadata ──────────────────────────────────────────────────────────────────

def encode_metadata_request() -> bytes:
    hdr = msgpack.packb({"type": "metadata"})
    return struct.pack("<I", len(hdr)) + hdr


def encode_metadata_response(data: dict) -> bytes:
    hdr = msgpack.packb({"type": "metadata_response", "data": data})
    return struct.pack("<I", len(hdr)) + hdr


def decode_metadata_response(raw: bytes) -> dict:
    (hlen,) = struct.unpack_from("<I", raw, 0)
    header = msgpack.unpackb(raw[4:4 + hlen], raw=False)
    return header.get("data", {})


# ── reset ─────────────────────────────────────────────────────────────────────

def encode_reset() -> bytes:
    hdr = msgpack.packb({"type": "reset"})
    return struct.pack("<I", len(hdr)) + hdr


def encode_reset_response(obs: Observation, info: dict) -> bytes:
    return _encode_obs_frame("reset_response", obs, {"info": info})


def decode_reset_response(data: bytes) -> tuple[Observation, dict]:
    obs, header = _decode_obs_frame(data)
    return obs, header.get("info", {})


# ── obs_request / obs_response ────────────────────────────────────────────────

def encode_obs_request() -> bytes:
    hdr = msgpack.packb({"type": "obs_request"})
    return struct.pack("<I", len(hdr)) + hdr


def encode_obs_response(obs: Observation) -> bytes:
    return _encode_obs_frame("obs_response", obs, {})


def decode_obs_response(data: bytes) -> Observation:
    obs, _ = _decode_obs_frame(data)
    return obs


# ── apply_action (fire-and-forget) ────────────────────────────────────────────

def encode_apply_action(
    action: np.ndarray,
    obs_timestamps: dict[str, float] | None = None,
) -> bytes:
    action = np.asarray(action, dtype=np.float32)
    header = msgpack.packb({
        "type": "apply_action",
        "shape": list(action.shape),
        "dtype": action.dtype.name,
        "obs_timestamps": obs_timestamps or {},
    })
    return struct.pack("<I", len(header)) + header + action.tobytes()


def decode_apply_action(data: bytes) -> tuple[np.ndarray, dict[str, float]]:
    (hlen,) = struct.unpack_from("<I", data, 0)
    header = msgpack.unpackb(data[4:4 + hlen], raw=False)
    arr = np.frombuffer(data[4 + hlen:], dtype=header["dtype"]).reshape(header["shape"]).copy()
    obs_timestamps = {k: float(v) for k, v in header.get("obs_timestamps", {}).items()}
    return arr, obs_timestamps
