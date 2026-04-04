"""Example policy client using Zenoh transport and the streaming (decoupled) API.

Start zenoh_server_example.py first, then run this:

    uv run examples/python/zenoh_server_example.py &
    uv run examples/python/zenoh_client_example.py
"""
import time
import threading

import numpy as np
import chiral

ACTION_HZ    = 10
OBS_HZ       = 30
CHUNK_SIZE   = 10
DOF          = 7
TOTAL_STEPS  = 200


def policy_loop(env: chiral.PolicyClient, stop: threading.Event) -> None:
    inference_count = 0
    while not stop.is_set():
        obs = env.latest_obs
        if obs is None:
            time.sleep(0.01)
            continue

        cam_info = ", ".join(
            f"{c.name}@{c.timestamp:.3f}s img={c.image.shape}"
            for c in obs.cameras[:2]
        )
        print(f"[policy #{inference_count}] obs ts={obs.timestamp:.3f}  cameras: {cam_info}",
              flush=True)

        # ── policy inference goes here ─────────────────────────────────────
        # actions = model.predict(obs)  # shape (CHUNK_SIZE, DOF)
        actions = np.zeros((CHUNK_SIZE, DOF), dtype=np.float32)
        # ──────────────────────────────────────────────────────────────────

        for a in actions:
            env.put_action(a)

        inference_count += 1
        time.sleep(CHUNK_SIZE / ACTION_HZ * 0.9)


if __name__ == "__main__":
    with chiral.PolicyClient("tcp/localhost:7447", protocol="zenoh") as env:
        meta = env.get_metadata()
        print(f"cameras: {meta.get('cameras', [])}  "
              f"action_shape: {meta.get('action_shape', [1, DOF])}\n",
              flush=True)

        obs, info = env.reset()

        env.start_obs_stream(hz=OBS_HZ)
        env.start_action_dispatch(hz=ACTION_HZ)

        stop = threading.Event()
        policy_thread = threading.Thread(target=policy_loop, args=(env, stop), daemon=True)
        policy_thread.start()

        t_start = time.perf_counter()
        dispatched = 0
        while dispatched < TOTAL_STEPS:
            time.sleep(1 / ACTION_HZ)
            dispatched += 1

        stop.set()
        policy_thread.join()

        elapsed = time.perf_counter() - t_start
        print(f"\ndispatched {TOTAL_STEPS} actions in {elapsed:.1f}s  "
              f"({TOTAL_STEPS / elapsed:.1f} Hz)")
