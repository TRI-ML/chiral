import asyncio
import threading
from collections import deque

import numpy as np
import websockets
import zenoh

from .protocol import (
    decode_metadata_response,
    decode_obs_response,
    decode_reset_response,
    encode_apply_action,
    encode_metadata_request,
    encode_obs_request,
    encode_reset,
)
from .types import Observation


class PolicyClient:
    """
    Client — policy/inference side.

    Typical usage with chunked policy predictions::

        with PolicyClient("ws://localhost:8765") as env:
            obs, info = env.reset()
            env.start_obs_stream(hz=30)       # thread 1: keeps latest_obs fresh
            env.start_action_dispatch(hz=10)  # thread 3: drains action queue at 10 Hz

            while True:                       # thread 2: policy inference
                obs = env.latest_obs
                if obs is None:
                    continue
                actions = policy(obs)         # shape (N, D) — chunked predictions
                for a in actions:
                    env.put_action(a)

    Pass ``protocol="zenoh"`` to use Zenoh pub/sub instead of WebSocket.
    ``uri`` is ``"ws://host:port"`` for WebSocket and ``"tcp/host:port"``
    for Zenoh.
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

        self._latest_obs: Observation | None = None

        # Streaming state
        self._obs_task: asyncio.Task | None = None
        self._action_task: asyncio.Task | None = None
        self._action_queue: deque = deque()

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
        # Stop any running streaming tasks first.
        if self._obs_task is not None or self._action_task is not None:
            asyncio.run_coroutine_threadsafe(
                self._astop_streams(), self._loop
            ).result()

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

    # ── API ───────────────────────────────────────────────────────────────────

    def get_metadata(self) -> dict:
        """Fetch static metadata from the server (action shape, camera names, etc.)."""
        return asyncio.run_coroutine_threadsafe(self._aget_metadata(), self._loop).result()

    def reset(self) -> tuple[Observation, dict]:
        """Send reset request and return (observation, info)."""
        return asyncio.run_coroutine_threadsafe(self._areset(), self._loop).result()

    @property
    def latest_obs(self) -> Observation | None:
        """Most recent observation received by the obs stream thread.

        Returns ``None`` until the first obs arrives after ``start_obs_stream``
        (or after ``reset``).
        """
        return self._latest_obs

    def get_obs(self) -> Observation:
        """Request and return the current observation (single blocking call).

        Useful for a one-shot fetch outside of the background stream.
        """
        return asyncio.run_coroutine_threadsafe(self._aget_obs(), self._loop).result()

    def start_obs_stream(self, hz: float = 30.0) -> None:
        """Start a background coroutine that polls the server for observations at *hz* Hz.

        Each response is stored in ``latest_obs``. Returns immediately.
        Call ``stop_obs_stream()`` (or ``close()``) to stop.
        """
        asyncio.run_coroutine_threadsafe(
            self._astart_obs_stream(hz), self._loop
        ).result()

    def stop_obs_stream(self) -> None:
        """Stop the obs stream background coroutine and wait for it to finish."""
        asyncio.run_coroutine_threadsafe(self._astop_obs_stream(), self._loop).result()

    def put_action(self, action: np.ndarray) -> None:
        """Enqueue a single action for dispatch.

        Thread-safe. Call from the policy/inference thread. Actions are
        sent FIFO by the dispatch thread at the configured Hz.
        """
        self._action_queue.append(action)

    def start_action_dispatch(self, hz: float = 10.0) -> None:
        """Start a background coroutine that sends queued actions at *hz* Hz.

        At each tick the coroutine pops one action from the front of the
        queue (if any) and sends it as a fire-and-forget message — the
        server applies it via ``apply_action()`` with no response, so obs
        polling is never blocked.

        Returns immediately. Call ``stop_action_dispatch()`` (or
        ``close()``) to stop.
        """
        asyncio.run_coroutine_threadsafe(
            self._astart_action_dispatch(hz), self._loop
        ).result()

    def stop_action_dispatch(self) -> None:
        """Stop the action dispatch background coroutine and wait for it to finish."""
        asyncio.run_coroutine_threadsafe(
            self._astop_action_dispatch(), self._loop
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
        obs, info = decode_reset_response(await self._recv())
        self._latest_obs = obs
        return obs, info

    async def _aget_obs(self) -> Observation:
        await self._send(encode_obs_request())
        obs = decode_obs_response(await self._recv())
        self._latest_obs = obs
        return obs

    # ── streaming coroutines ──────────────────────────────────────────────────

    async def _aobs_stream(self, hz: float) -> None:
        interval = 1.0 / hz
        while True:
            t0 = self._loop.time()
            await self._send(encode_obs_request())
            data = await self._recv()
            self._latest_obs = decode_obs_response(data)
            elapsed = self._loop.time() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _aaction_dispatch(self, hz: float) -> None:
        interval = 1.0 / hz
        while True:
            t0 = self._loop.time()
            if self._action_queue:
                action = self._action_queue.popleft()
                obs_timestamps = (
                    {cam.name: cam.timestamp for cam in self._latest_obs.cameras}
                    if self._latest_obs is not None else {}
                )
                await self._send(encode_apply_action(action, obs_timestamps))
            elapsed = self._loop.time() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _astart_obs_stream(self, hz: float) -> None:
        self._obs_task = asyncio.ensure_future(self._aobs_stream(hz))

    async def _astart_action_dispatch(self, hz: float) -> None:
        self._action_task = asyncio.ensure_future(self._aaction_dispatch(hz))

    async def _astop_obs_stream(self) -> None:
        if self._obs_task and not self._obs_task.done():
            self._obs_task.cancel()
            try:
                await self._obs_task
            except asyncio.CancelledError:
                pass
        self._obs_task = None

    async def _astop_action_dispatch(self) -> None:
        if self._action_task and not self._action_task.done():
            self._action_task.cancel()
            try:
                await self._action_task
            except asyncio.CancelledError:
                pass
        self._action_task = None

    async def _astop_streams(self) -> None:
        await self._astop_obs_stream()
        await self._astop_action_dispatch()
