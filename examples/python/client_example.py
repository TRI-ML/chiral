"""Example policy client.

Start server_example.py first, then run this:

    uv run examples/python/server_example.py &
    uv run examples/python/client_example.py
"""
import time
from collections import deque

import numpy as np
from tqdm import tqdm
import chiral

WINDOW = 20  # steps used for rolling stats


if __name__ == "__main__":
    with chiral.PolicyClient("ws://localhost:8765") as env:
        meta = env.get_metadata()
        cameras      = meta.get("cameras", [])
        action_shape = meta.get("action_shape", [1, 7])
        print(f"cameras: {cameras}  action_shape: {action_shape}\n", flush=True)

        obs, info = env.reset()
        latencies: list[float] = []
        window = deque(maxlen=WINDOW)
        step   = 0
        t_episode = time.perf_counter()

        with tqdm(unit="step", dynamic_ncols=True) as pbar:
            while True:
                action = np.zeros(action_shape, dtype=np.float32)

                t0 = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                latency_ms = (time.perf_counter() - t0) * 1e3

                step += 1
                latencies.append(latency_ms)
                window.append(latency_ms)
                fps = step / (time.perf_counter() - t_episode)

                w = np.array(window)
                pbar.set_postfix(
                    lat=f"{latency_ms:.0f}ms",
                    mean=f"{w.mean():.0f}ms",
                    p95=f"{np.percentile(w, 95):.0f}ms",
                    fps=f"{fps:.1f}",
                )
                pbar.update(1)

                if terminated or truncated:
                    break

        arr = np.array(latencies)
        fps = step / (time.perf_counter() - t_episode)
        print(f"\n── episode summary ──────────────────────────────")
        print(f"steps={step}  avg_fps={fps:.1f}")
        print(f"mean={arr.mean():.1f}ms  median={np.median(arr):.1f}ms")
        print(f"min={arr.min():.1f}ms   max={arr.max():.1f}ms")
        print(f"p95={np.percentile(arr, 95):.1f}ms  p99={np.percentile(arr, 99):.1f}ms")
