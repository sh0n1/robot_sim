import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from distance_transform_cuda import distance_transform_cuda  # noqa: E402
from test_correctness import ARENA, GOAL, load_reference, make_grid  # noqa: E402


def timed(fn, repeats):
    out, times = None, []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, times


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[101, 251, 501, 1001, 2001])
    ap.add_argument("--repeats", type=int, default=5, help="GPU repeats per N")
    ap.add_argument("--cpu-repeats", type=int, default=3, help="CPU repeats for N <= 501")
    ap.add_argument("--cpu-max-n", type=int, default=2001, help="largest N to run the CPU version at")
    ap.add_argument("--check-every", type=int, default=16)
    ap.add_argument("--float32", action="store_true", help="use float32 on the GPU (faster, ~1e-4 accuracy)")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    dtype = np.float32 if args.float32 else np.float64
    create_obstacle_grid, cpu_dt = load_reference()

    from numba import cuda
    if os.environ.get("NUMBA_ENABLE_CUDASIM") != "1":
        print("GPU:", cuda.get_current_device().name.decode())

    # Warm-up: JIT-compile the kernel and create the CUDA context (not timed).
    warm = make_grid(create_obstacle_grid, 51)
    distance_transform_cuda(warm, GOAL, 51, arena_size=ARENA, check_every=args.check_every, dtype=dtype)

    rows = []
    for n in args.n:
        grid = make_grid(create_obstacle_grid, n)
        row = {"N": n, "cells": n * n, "cpu_s": "", "gpu_total_s": "", "kernels_s": "",
               "upload_s": "", "download_s": "", "rounds": "", "speedup": "", "max_abs_err": ""}

        ref = None
        if n <= args.cpu_max_n:
            reps = args.cpu_repeats if n <= 501 else 1
            ref, t_cpu = timed(lambda: cpu_dt(grid, GOAL, n, arena_size=ARENA), reps)
            row["cpu_s"] = statistics.mean(t_cpu)

        stats_runs, totals = [], []
        got = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            got, st = distance_transform_cuda(grid, GOAL, n, arena_size=ARENA, check_every=args.check_every,
                                              dtype=dtype, return_stats=True)
            totals.append(time.perf_counter() - t0)
            stats_runs.append(st)
        row["gpu_total_s"] = statistics.mean(totals)
        row["kernels_s"] = statistics.mean(s["kernels_s"] for s in stats_runs)
        row["upload_s"] = statistics.mean(s["upload_s"] for s in stats_runs)
        row["download_s"] = statistics.mean(s["download_s"] for s in stats_runs)
        row["rounds"] = stats_runs[0]["rounds"]

        if ref is not None:
            finite = np.isfinite(ref)
            same = np.array_equal(finite, np.isfinite(got))
            row["max_abs_err"] = float(np.max(np.abs(ref[finite] - got[finite]))) if same else float("inf")
            row["speedup"] = row["cpu_s"] / row["gpu_total_s"]

        rows.append(row)
        sp = f"{row['speedup']:.1f}x" if row["speedup"] != "" else "n/a"
        cpu = f"{row['cpu_s']:.3f}s" if row["cpu_s"] != "" else "skipped"
        print(f"N={n:5d}  cpu={cpu:>9}  gpu={row['gpu_total_s']:.4f}s "
              f"(kernels {row['kernels_s']:.4f}s, up {row['upload_s']:.4f}s, down {row['download_s']:.4f}s)  "
              f"rounds={row['rounds']}  speedup={sp}  err={row['max_abs_err']}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns = [r["N"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        cpu_pts = [(r["N"], r["cpu_s"]) for r in rows if r["cpu_s"] != ""]
        ax[0].loglog([p[0] for p in cpu_pts], [p[1] for p in cpu_pts], "o-", label="Sequential (CPU, FIFO queue)")
        ax[0].loglog(ns, [r["gpu_total_s"] for r in rows], "s-", label="CUDA (end to end)")
        ax[0].set_xlabel("Grid size N (N x N cells)")
        ax[0].set_ylabel("Time (s)")
        ax[0].set_title("Distance transform time")
        ax[0].grid(True, which="both", alpha=0.3)
        ax[0].legend()
        sp_pts = [(r["N"], r["speedup"]) for r in rows if r["speedup"] != ""]
        ax[1].plot([p[0] for p in sp_pts], [p[1] for p in sp_pts], "o-")
        ax[1].set_xlabel("Grid size N")
        ax[1].set_ylabel("Speed-up (CPU time / GPU time)")
        ax[1].set_title("Speed-up")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig("results.png", dpi=150)
        print("wrote results.png")
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
