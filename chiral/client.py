import asyncio
import threading

import numpy as np
import websockets
import zenoh

from .protocol import (
    decode_metadata_response,
    decode_reset_response,
    decode_step_response,
    encode_action,
    encode_metadata_request,
    encode_reset,
)
from .types import Observation


class PolicyClient:
    """
    Client — policy/inference side, gym-like interface.

    Pass ``protocol="zenoh"`` to use Zenoh pub/sub instead of WebSocket.
    The ``uri`` argument is interpreted according to the protocol:
    WebSocket expects ``"ws://host:port"``; Zenoh expects a locator such
    as ``"tcp/host:port"``.

    Usage::

        # WebSocket (default)
        with PolicyClient("ws://localhost:8765") as env:
            obs, info = env.reset()
            ...

        # Zenoh
        with PolicyClient("tcp/localhost:7447", protocol="zenoh") as env:
            obs, info = env.reset()
            ...
    """

    def __init__(
        self,
        uri: str | None = None,
        protocol: str = "websocket",
    ) -> None:
        if uri is None:
            uri = "ws://localhost:8765" if protocol == "websocket" else "tcp/localhost:7447"
        self.uri = uri
        self._protocol = protocol

        self._loop = asyncio.new_event_loop()
        self._ws = None
        # Zenoh state
        self._session = None
        self._pub = None
        self._sub = None
        self._response_queue: asyncio.Queue = asyncio.Queue()

        self._last_obs: Observation | None = None
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self) -> "PolicyClient":
        asyncio.run_coroutine_threadsafe(self._aconnect(), self._loop).result()
        return self

    async def _aconnect(self) -> None:
        if self._protocol == "zenoh":
            await self._aconnect_zenoh()
        else:
            await self._aconnect_ws()

    async def _aconnect_ws(self) -> None:
        warned = False
        while True:
            try:
                self._ws = await websockets.connect(
                    self.uri, max_size=None, compression=None
                )
                return
            except OSError:
                if not warned:
                    print(f"chiral: waiting for server at {self.uri} …", flush=True)
                    warned = True
                await asyncio.sleep(0.1)

    def _on_zenoh_response(self, sample: zenoh.Sample) -> None:
        data = bytes(sample.payload)
        asyncio.run_coroutine_threadsafe(
            self._response_queue.put(data), self._loop
        )

    async def _aconnect_zenoh(self) -> None:
        config = zenoh.Config()
        config.insert_json5("connect/endpoints", f'["{self.uri}"]')

        self._session = zenoh.open(config)
        self._pub = self._session.declare_publisher("chiral/c2s")
        self._sub = self._session.declare_subscriber("chiral/s2c", self._on_zenoh_response)

        # Probe until the server responds, ensuring Zenoh routes are established.
        warned = False
        while True:
            while not self._response_queue.empty():
                self._response_queue.get_nowait()
            self._pub.put(encode_metadata_request())
            try:
                await asyncio.wait_for(self._response_queue.get(), timeout=0.5)
                return
            except asyncio.TimeoutError:
                if not warned:
                    print(f"chiral: waiting for Zenoh server at {self.uri} …", flush=True)
                    warned = True

    def close(self) -> None:
        if self._protocol == "zenoh":
            if self._session:
                self._session.close()
                self._session = None
        else:
            if self._ws:
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop).result()
                self._ws = None
        self._loop.call_soon_threadsafe(self._loop.stop)

    def __enter__(self) -> "PolicyClient":
        return self.connect()

    def __exit__(self, *args) -> None:
        self.close()

    # ── gym-like API ──────────────────────────────────────────────────────────

    def get_metadata(self) -> dict:
        """Fetch static metadata from the server (action shape, camera names, etc.)."""
        return asyncio.run_coroutine_threadsafe(self._aget_metadata(), self._loop).result()

    def reset(self) -> tuple[Observation, dict]:
        """Send reset request and return (observation, info)."""
        return asyncio.run_coroutine_threadsafe(self._areset(), self._loop).result()

    def step(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, bool, dict]:
        """Send action and return (obs, reward, terminated, truncated, info)."""
        return asyncio.run_coroutine_threadsafe(
            self._astep(action), self._loop
        ).result()

    # ── async internals ───────────────────────────────────────────────────────

    async def _send(self, data: bytes) -> None:
        if self._protocol == "zenoh":
            self._pub.put(data)
        else:
            await self._ws.send(data)

    async def _recv(self) -> bytes:
        if self._protocol == "zenoh":
            return await self._response_queue.get()
        raw = await self._ws.recv()
        return raw if isinstance(raw, bytes) else raw.encode()

    async def _aget_metadata(self) -> dict:
        await self._send(encode_metadata_request())
        return decode_metadata_response(await self._recv())

    async def _areset(self) -> tuple[Observation, dict]:
        await self._send(encode_reset())
        result = decode_reset_response(await self._recv())
        self._last_obs = result[0]
        return result

    async def _astep(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, bool, dict]:
        obs_timestamps = (
            {cam.name: cam.timestamp for cam in self._last_obs.cameras}
            if self._last_obs is not None else {}
        )
        await self._send(encode_action(action, obs_timestamps))
        result = decode_step_response(await self._recv())
        self._last_obs = result[0]
        return result
