# Python Examples — Zenoh

The Zenoh example scripts in `examples/python/` are drop-in counterparts to the WebSocket examples. The server and client interfaces are identical; only the transport changes.

```bash
uv run examples/python/zenoh_server_example.py &
uv run examples/python/zenoh_client_example.py
```

---

## What is different from the WebSocket examples

| | WebSocket | Zenoh |
|---|---|---|
| Server constructor | `super().__init__(host="0.0.0.0", port=8765)` | `super().__init__(protocol="zenoh")` |
| Client URI | `"ws://localhost:8765"` | `"tcp/localhost:7447"` |
| Default port | 8765 | 7447 |
| Transport | HTTP upgrade → WebSocket frames | Zenoh binary framing over TCP |

Everything else — `camera_configs`, `proprio_configs`, `reset`, `step`, the `update_*` helpers, and the tqdm latency measurement on the client — is unchanged.

---

## Server

**File:** `examples/python/zenoh_server_example.py`

The only difference from `server_example.py` is the constructor:

```python
class MyRobotServer(chiral.PolicyServer):
    def __init__(self):
        super().__init__(protocol="zenoh")  # listens on tcp/0.0.0.0:7447
        ...
```

`protocol="zenoh"` selects the Zenoh transport. The default port when using Zenoh is `7447`. To use a different port:

```python
super().__init__(host="0.0.0.0", port=7448, protocol="zenoh")
```

---

## Client

**File:** `examples/python/zenoh_client_example.py`

The only difference from `client_example.py` is the URI and protocol argument:

```python
with chiral.PolicyClient("tcp/localhost:7447", protocol="zenoh") as env:
    ...
```

`"tcp/localhost:7447"` is also the default when `protocol="zenoh"` is set and no URI is given, so on localhost this can be shortened to:

```python
with chiral.PolicyClient(protocol="zenoh") as env:
    ...
```

---

## Benchmarking

Run the WebSocket and Zenoh examples back-to-back to compare latency under identical conditions. The tqdm summary at the end of each client run prints mean, median, min, max, p95, and p99 latency across the full episode, making it straightforward to compare the two transports.

```bash
# WebSocket
uv run examples/python/server_example.py &
uv run examples/python/client_example.py
kill %1

# Zenoh
uv run examples/python/zenoh_server_example.py &
uv run examples/python/zenoh_client_example.py
kill %1
```
