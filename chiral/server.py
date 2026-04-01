import asyncio
import threading
import time
from abc import ABC, abstractmethod

import numpy as np
import websockets
import zenoh

from .protocol import (
    decode_action,
    encode_metadata_response,
    encode_reset_response,
    encode_step_response,
    peek_type,
)
from .types import CameraConfig, CameraInfo, Observation, ProprioConfig


class PolicyServer(ABC):
    """
    Abstract server — robot/environment side.

    Subclass and implement ``camera_configs``, ``reset``, and ``step``,
    then call ``run()`` (blocking) or ``await serve()`` (async).

    The base class calls ``camera_configs()`` once at init time and
    pre-allocates ``self.images`` and ``self.depths`` so that sensor
    drivers can fill them in-place without per-step allocation.

    The client drives the loop: it sends a reset request first, then
    repeatedly sends actions and receives step responses.

    Pass ``protocol="zenoh"`` to use Zenoh pub/sub instead of WebSocket.
    The ``host`` and ``port`` arguments are used for both transports:
    WebSocket listens on ``ws://host:port``; Zenoh listens on
    ``tcp/host:port`` (default port 7447 when Zenoh is selected).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        protocol: str = "websocket",
    ) -> None:
        self.host = host
        self.port = port if port is not None else (8765 if protocol == "websocket" else 7447)
        self._protocol = protocol

        configs = self.camera_configs()
        self._configs: list[CameraConfig] = configs

        # Pre-allocated buffers — write via update_* helpers from sensor threads;
        # read via _make_obs() on the network thread.
        self.images: dict[str, np.ndarray] = {
            c.name: np.zeros((c.height, c.width, c.channels), dtype=c.image_dtype)
            for c in configs
        }
        self.depths: dict[str, np.ndarray] = {
            c.name: np.zeros((c.height, c.width), dtype=c.depth_dtype)
            for c in configs
            if c.has_depth
        }
        # Intrinsics/extrinsics are mutable per step (e.g. wrist camera moves
        # with the arm). Initialized from CameraConfig; update each step via
        # update_intrinsics() / update_extrinsics().
        self.intrinsics: dict[str, np.ndarray] = {
            c.name: c.intrinsics.copy() for c in configs
        }
        self.extrinsics: dict[str, np.ndarray] = {
            c.name: c.extrinsics.copy() for c in configs
        }

        # Monotonic timestamp of the latest frame written via update_image().
        self._image_timestamps: dict[str, float] = {c.name: 0.0 for c in configs}

        # One lock per camera protects images, depths, intrinsics, extrinsics, and timestamp.
        self._locks: dict[str, threading.Lock] = {
            c.name: threading.Lock() for c in configs
        }

        proprio_configs = self.proprio_configs()
        self._proprio_configs: list[ProprioConfig] = proprio_configs

        # Pre-allocated proprioception buffers.
        self.proprios: dict[str, np.ndarray] = {
            p.name: np.zeros(p.size, dtype=p.dtype)
            for p in proprio_configs
        }
        self._proprio_locks: dict[str, threading.Lock] = {
            p.name: threading.Lock() for p in proprio_configs
        }

        # Zenoh state (initialised in serve() when protocol="zenoh").
        self._zenoh_request_queue: asyncio.Queue | None = None
        self._zenoh_loop: asyncio.AbstractEventLoop | None = None

    @abstractmethod
    def camera_configs(self) -> list[CameraConfig]:
        """Return the list of cameras this server will stream.

        Called once during ``__init__`` to pre-allocate ``self.images``
        and ``self.depths``.
        """

    def proprio_configs(self) -> list[ProprioConfig]:
        """Return the list of proprioception streams. Default: none.

        Override to stream proprio data alongside images. Called once
        during ``__init__`` to pre-allocate ``self.proprios``.
        """
        return []

    def update_image(self, name: str, data: np.ndarray) -> None:
        """Copy *data* into the pre-allocated image buffer for *name*.

        Records ``time.monotonic()`` as the frame timestamp.
        Safe to call from any thread concurrently with other cameras.
        """
        with self._locks[name]:
            np.copyto(self.images[name], data)
            self._image_timestamps[name] = time.monotonic()

    def update_depth(self, name: str, data: np.ndarray) -> None:
        """Copy *data* into the pre-allocated depth buffer for *name*.

        Safe to call from any thread concurrently with other cameras.
        """
        with self._locks[name]:
            np.copyto(self.depths[name], data)

    def update_intrinsics(self, name: str, matrix: np.ndarray) -> None:
        """Update the (3, 3) intrinsics matrix for *name* (e.g. zoom lens).

        Safe to call from any thread concurrently with other cameras.
        """
        with self._locks[name]:
            np.copyto(self.intrinsics[name], matrix)

    def update_extrinsics(self, name: str, matrix: np.ndarray) -> None:
        """Update the (4, 4) camera-to-world extrinsics for *name*.

        Call this every step for cameras that move (e.g. wrist cameras).
        Safe to call from any thread concurrently with other cameras.
        """
        with self._locks[name]:
            np.copyto(self.extrinsics[name], matrix)

    def update_proprio(self, name: str, data: np.ndarray) -> None:
        """Copy *data* into the pre-allocated proprio buffer for *name*.

        Safe to call from any thread concurrently with other streams.
        """
        with self._proprio_locks[name]:
            np.copyto(self.proprios[name], data)

    def _make_obs(self, timestamp: float = 0.0) -> Observation:
        """Snapshot all buffers under their per-camera/proprio locks and return an Observation."""
        cameras = []
        for c in self._configs:
            with self._locks[c.name]:
                image      = self.images[c.name].copy()
                depth      = self.depths[c.name].copy() if c.name in self.depths else None
                intrinsics = self.intrinsics[c.name].copy()
                extrinsics = self.extrinsics[c.name].copy()
                cam_ts     = self._image_timestamps[c.name]
            cameras.append(
                CameraInfo(
                    name=c.name,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    image=image,
                    depth=depth,
                    timestamp=cam_ts,
                )
            )

        proprios = {}
        for p in self._proprio_configs:
            with self._proprio_locks[p.name]:
                proprios[p.name] = self.proprios[p.name].copy()

        return Observation(cameras=cameras, proprios=proprios, timestamp=timestamp)

    async def get_metadata(self) -> dict:
        """Return static environment metadata (action shape, camera names, etc.).
        Override to expose information; default returns an empty dict."""
        return {}

    @abstractmethod
    async def reset(self) -> tuple[Observation, dict]:
        """Reset the environment and return (observation, info)."""

    @abstractmethod
    async def step(
        self, action: np.ndarray
    ) -> tuple[Observation, float, bool, bool, dict]:
        """Step the environment and return (obs, reward, terminated, truncated, info)."""

    # ── WebSocket transport ───────────────────────────────────────────────────

    async def _handle(self, websocket) -> None:
        try:
            while True:
                raw = await websocket.recv()
                if not isinstance(raw, bytes):
                    raw = raw.encode()

                msg_type = peek_type(raw)

                if msg_type == "metadata":
                    data = await self.get_metadata()
                    await websocket.send(encode_metadata_response(data))

                elif msg_type == "reset":
                    obs, info = await self.reset()
                    await websocket.send(encode_reset_response(obs, info))

                elif msg_type == "action":
                    action, _ = decode_action(raw)
                    obs, reward, terminated, truncated, info = await self.step(action)
                    await websocket.send(
                        encode_step_response(obs, reward, terminated, truncated, info)
                    )

        except websockets.ConnectionClosed:
            pass

    # ── Zenoh transport ───────────────────────────────────────────────────────

    def _on_zenoh_request(self, sample: zenoh.Sample) -> None:
        data = bytes(sample.payload)
        asyncio.run_coroutine_threadsafe(
            self._zenoh_request_queue.put(data), self._zenoh_loop
        )

    async def _zenoh_serve(self) -> None:
        self._zenoh_loop = asyncio.get_running_loop()
        self._zenoh_request_queue = asyncio.Queue()

        config = zenoh.Config()
        config.insert_json5("listen/endpoints", f'["tcp/{self.host}:{self.port}"]')

        session = zenoh.open(config)
        pub = session.declare_publisher("chiral/s2c")
        sub = session.declare_subscriber("chiral/c2s", self._on_zenoh_request)

        try:
            while True:
                raw = await self._zenoh_request_queue.get()
                msg_type = peek_type(raw)

                if msg_type == "metadata":
                    pub.put(encode_metadata_response(await self.get_metadata()))

                elif msg_type == "reset":
                    obs, info = await self.reset()
                    pub.put(encode_reset_response(obs, info))

                elif msg_type == "action":
                    action, _ = decode_action(raw)
                    obs, reward, terminated, truncated, info = await self.step(action)
                    pub.put(encode_step_response(obs, reward, terminated, truncated, info))

        finally:
            sub.undeclare()
            pub.undeclare()
            session.close()

    # ── dispatch ──────────────────────────────────────────────────────────────

    async def serve(self) -> None:
        if self._protocol == "zenoh":
            await self._zenoh_serve()
        else:
            async with websockets.serve(self._handle, self.host, self.port,
                                        max_size=None, compression=None):
                await asyncio.Future()

    def run(self) -> None:
        asyncio.run(self.serve())
