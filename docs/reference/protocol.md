# Protocol Reference

Both the Python and C++ implementations produce and consume identical byte sequences, enabling cross-language pairing.

---

## Frame Structure

Every message sent over the WebSocket connection — in either direction — uses the same framing:

```
 ┌────────────────────┬─────────────────────────┬────────────────────────┐
 │  header_len (4 B)  │  msgpack header          │  raw payload bytes     │
 │  little-endian     │  (header_len bytes)      │  (remainder of frame)  │
 │  uint32            │                          │                        │
 └────────────────────┴─────────────────────────┴────────────────────────┘
```

1. **4-byte header length prefix** — a little-endian `uint32` giving the number of bytes in the following msgpack header. This allows the receiver to split the frame without scanning for delimiters.
2. **msgpack header** — a MessagePack map containing the message type and all metadata fields (camera names, shapes, dtypes, byte offsets, etc.). No pixel data or float buffers live here.
3. **raw payload** — the bulk data: image bytes, depth bytes, and proprio floats concatenated in declaration order. Camera and proprio entries in the msgpack header include `offset` and `size` fields that index into this payload.

### Python framing code

```python
import struct, msgpack

# Encode a frame
header = msgpack.packb({"type": "reset", ...})
payload = b"".join(raw_buffers)
frame = struct.pack("<I", len(header)) + header + payload

# Decode a frame
(hlen,) = struct.unpack_from("<I", data, 0)
header  = msgpack.unpackb(data[4 : 4 + hlen], raw=False)
payload = data[4 + hlen :]
```

---

## Message Types

There are six message types in total. Three are sent from the client to the server; three are sent from the server to the client.

| Direction | Type | msgpack header | Payload |
|-----------|------|----------------|---------|
| Client → Server | `metadata` | `{"type": "metadata"}` | _(empty)_ |
| Client → Server | `reset` | `{"type": "reset"}` | _(empty)_ |
| Client → Server | `action` | `{"type": "action", "shape": [N, D], "dtype": "float32"}` | N×D float32 values (row-major) |
| Server → Client | `metadata_response` | `{"type": "metadata_response", "data": {...}}` | _(empty)_ |
| Server → Client | `reset_response` | observation fields + `{"info": {...}}` | images + depths + proprios |
| Server → Client | `step_response` | observation fields + `{"reward": f, "terminated": b, "truncated": b, "info": {...}}` | images + depths + proprios |

### metadata / metadata_response

The client sends `{"type": "metadata"}` with no payload. The server responds with a msgpack map that includes the `data` key, which maps to the dict returned by the server's `get_metadata()` override:

```json
{"type": "metadata_response", "data": {"cameras": ["wrist_cam"], "action_shape": [1, 7]}}
```

### reset / reset_response

The client sends `{"type": "reset"}` with no payload. The server calls its `reset()` method and responds with a full observation frame (see below) whose type is `"reset_response"`.

### action / step_response

The client sends an action:

```
header: {"type": "action", "shape": [1, 7], "dtype": "float32"}
payload: 7 × 4 = 28 bytes of float32 data
```

The server decodes the action, calls `step()`, and responds with a full observation frame of type `"step_response"`.

---

## Observation Frame Header

Both `reset_response` and `step_response` use the same observation frame structure. The msgpack header for an observation frame contains these top-level keys:

| Key | Type | Description |
|-----|------|-------------|
| `type` | `string` | `"reset_response"` or `"step_response"` |
| `timestamp` | `float` | Application timestamp set by `make_obs(timestamp)` |
| `cameras` | `array` | One entry per camera (see below) |
| `proprios` | `array` | One entry per proprio stream (see below) |
| `extra` | `map` | Extra fields from `Observation.extra` (Python only; empty in C++) |
| `info` | `map` | Step info dict from `reset()` / `step()` return value |
| `reward` | `float` | Scalar reward _(step_response only)_ |
| `terminated` | `bool` | Episode terminated flag _(step_response only)_ |
| `truncated` | `bool` | Episode truncated flag _(step_response only)_ |

### Camera Entries

Each element of the `cameras` array is a map with these fields:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `string` | Camera name matching `CameraConfig.name` |
| `intrinsics` | `array[9] float64` | Camera matrix K, flattened row-major from the 3×3 matrix |
| `extrinsics` | `array[16] float64` | Camera-to-world transform T, flattened row-major from the 4×4 matrix |
| `image_shape` | `array[3] int` | `[H, W, C]` |
| `image_dtype` | `string` | e.g. `"uint8"` |
| `image_offset` | `int` | Byte offset of image data within the payload |
| `image_size` | `int` | Byte length of image data |
| `depth_shape` | `array[2] int` | `[H, W]` — present only if has_depth |
| `depth_dtype` | `string` | e.g. `"float32"` — present only if has_depth |
| `depth_offset` | `int` | Byte offset of depth data within the payload — present only if has_depth |
| `depth_size` | `int` | Byte length of depth data — present only if has_depth |

!!! note "Intrinsics and extrinsics on every frame"
    The 9 intrinsics floats and 16 extrinsics floats are included in **every** observation header — not just on the first step. This is intentional: cameras mounted on robot arms change pose every frame, so the client must always read fresh values and never cache the previous step's values.

### Proprio Entries

Each element of the `proprios` array is a map with these fields:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `string` | Stream name matching `ProprioConfig.name` |
| `dtype` | `string` | `"float32"` |
| `offset` | `int` | Byte offset of proprio data within the payload |
| `size` | `int` | Byte length of proprio data (`n_elements * 4` for float32) |

### Payload Layout

Camera images come first (in declaration order), then camera depth maps (interleaved: image₀, depth₀, image₁, depth₁, …), then proprio buffers. More precisely, the payload is built by appending, for each camera in order: image bytes first, then depth bytes (if any); then for each proprio stream in order: float32 bytes.

The `offset` fields in the header are absolute byte offsets from the start of the payload region (i.e. from byte `4 + header_len` of the frame).

---

## No Compression

WebSocket `permessage-deflate` compression is **explicitly disabled** on both the server and the client:

```python
# Python — websockets library
websockets.serve(..., compression=None)    # server
websockets.connect(..., compression=None)  # client
```

```cpp
// C++ — ixwebsocket
webSocket.disablePerMessageDeflate();
```

Reasons:

1. **Image data is already incompressible.** Raw uint8 RGB frames and float32 depth maps have high entropy; deflate achieves negligible compression ratios while consuming significant CPU time.
2. **Latency over throughput.** Chiral is designed for low-latency control loops (20–50 Hz). Compression adds per-frame CPU overhead on both sides, which increases round-trip latency.
3. **Predictable timing.** Without compression, encoding and decoding times are proportional to buffer size and are easy to reason about.

---

## Cross-Language Compatibility

The wire format is identical for Python and C++. The msgpack header is the same byte sequence; the payload is the same raw bytes. This means:

- A **Python server** can talk to a **C++ client** without any configuration change.
- A **C++ server** can talk to a **Python client** without any configuration change.

The only practical difference is that C++ `InfoMap` values are always strings, while Python dicts can hold arbitrary msgpack types (integers, floats, lists, nested dicts). When a Python server returns a non-string value in `get_metadata()` or `step()` info, it arrives at a C++ client as a msgpack-encoded value serialized into a string. Parse it with `std::stoi`/`std::stof` or a msgpack decoder as needed.

---

## Example: step_response Frame

Below is a concrete example of a `step_response` frame for a server with one RGB+depth camera (`480×640×3`, `uint8`/`float32`) and one proprio stream (`joint_pos`, 7 elements):

```
Offset  Length  Contents
------  ------  --------
0       4       header_len = <N> (little-endian uint32)
4       N       msgpack map:
                  type        = "step_response"
                  timestamp   = 0.05
                  reward      = 0.0
                  terminated  = false
                  truncated   = false
                  info        = {}
                  cameras[0]:
                    name         = "wrist_cam"
                    intrinsics   = [600,0,320,0,600,240,0,0,1]   (9 float64)
                    extrinsics   = [1,0,0,0, 0,1,0,0, 0,0,1,0, x,y,z,1]  (16 float64)
                    image_shape  = [480, 640, 3]
                    image_dtype  = "uint8"
                    image_offset = 0
                    image_size   = 921600        (480*640*3)
                    depth_shape  = [480, 640]
                    depth_dtype  = "float32"
                    depth_offset = 921600
                    depth_size   = 1228800       (480*640*4)
                  proprios[0]:
                    name   = "joint_pos"
                    dtype  = "float32"
                    offset = 2150400             (921600 + 1228800)
                    size   = 28                  (7 * 4)
4+N     921600  image bytes (480*640*3 uint8)
4+N+921600  1228800  depth bytes (480*640 float32)
4+N+2150400  28  proprio bytes (7 float32)
Total: 4 + N + 2150428 bytes
```
