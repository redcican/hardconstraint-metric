"""Run independent jobs on all GPUs at once (project rule: always use both 4060 Ti cards).

    rows = run_on_gpus(fn, jobs, gpus=[0, 1])

fn(job, device) -> list[dict] runs in a worker process pinned to one GPU. Jobs are handed out
dynamically, so a slow job does not leave the other card idle. Results come back in job order.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import torch


def available_gpus():
    return list(range(torch.cuda.device_count())) or [None]


def _init(queue):
    global _DEVICE
    gid = queue.get()
    _DEVICE = torch.device(f"cuda:{gid}") if gid is not None else torch.device("cpu")
    if gid is not None:
        torch.cuda.set_device(_DEVICE)
        # WSL pages VRAM to host memory when a card is oversubscribed (very slow, and driver
        # "Failed to create GPU mapping" crashes). Cap each process so it OOMs cleanly instead.
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("TCFLOW_MEM_FRAC", "0.9")), _DEVICE)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _call(args):
    """A failing job is reported and returns no rows; it does not take down the other jobs."""
    import traceback
    fn, job = args
    try:
        return fn(job, _DEVICE)
    except Exception:
        print(f"JOB FAILED on {_DEVICE}: {str(job)[:200]}\n{traceback.format_exc()}", flush=True)
        return []
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # long-lived workers: do not carry cached blocks into the next job


def run_on_gpus(fn, jobs, gpus=None, procs_per_gpu=1):
    gpus = available_gpus() if gpus is None else gpus
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    slots = [g for g in gpus for _ in range(procs_per_gpu)]
    for g in slots:
        q.put(g)
    with ctx.Pool(len(slots), initializer=_init, initargs=(q,)) as pool:
        return pool.map(_call, [(fn, j) for j in jobs], chunksize=1)


def cached_job(fn, cache_dir, key):
    """Run fn() once; its (JSON-serialisable) result is stored at cache_dir/key.json and reused."""
    import json
    from pathlib import Path
    p = Path(cache_dir) / f"{key}.json"
    if p.exists():
        return json.loads(p.read_text())
    out = fn()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.rename(p)
    return out
