"""Check the CUDA distance transform against the sequential one.

Run on your laptop (real GPU):      python test_correctness.py
Run without a GPU (CUDA simulator): NUMBA_ENABLE_CUDASIM=1 python test_correctness.py --n 41

Put this file next to controllers.py (your repo root).  If controllers.py can't
be imported (missing cv2 etc.), the reference functions are pulled out of its
source text instead, so the test always uses YOUR sequential code.
"""
import argparse
import ast
import os
import sys
import time
from queue import Queue

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from distance_transform_cuda import distance_transform_cuda  # noqa: E402

GOAL = np.array([1.5, 1.5])
ARENA = 3.0
OBSTACLES = [(0.5, 0.9, 0.7, 1), (1.3, 1.7, 1.1, 1.4), (0.7, 1.0, 1.3, 1.6),
             (2, 2.7, 1.5, 2.5), (1.7, 2.5, 0.7, 1.3), (1.9, 2.3, 0.4, 0.9),
             (0.3, 0.7, 1.9, 2.3), (1.4, 1.8, 2.2, 2.6), (0.1, 0.5, 2.5, 2.9),
             (1.0, 1.3, 0.1, 0.5)]  # same layout as main.py


def load_reference():
    try:
        from controllers import create_obstacle_grid, distance_transform
        return create_obstacle_grid, distance_transform
    except Exception:  # cv2/matplotlib not installed, etc.
        src = open(os.path.join(HERE, "controllers.py")).read()
        wanted = {"create_obstacle_grid", "distance_transform"}
        funcs = [n for n in ast.parse(src).body
                 if isinstance(n, ast.FunctionDef) and n.name in wanted]
        ns = {"np": np, "Queue": Queue}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "controllers.py", "exec"), ns)
        return ns["create_obstacle_grid"], ns["distance_transform"]


def make_grid(create_obstacle_grid, n):
    buf = (n - 1) // 50
    grid, _, _ = create_obstacle_grid(OBSTACLES, n, buf, arena_size=ARENA)
    return grid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[101, 251])
    ap.add_argument("--variant", nargs="+", default=["sync", "inplace"], choices=["sync", "inplace"])
    args = ap.parse_args()

    create_obstacle_grid, cpu_dt = load_reference()
    ok = True
    for n in args.n:
        grid = make_grid(create_obstacle_grid, n)
        t0 = time.perf_counter()
        ref = cpu_dt(grid, GOAL, n, arena_size=ARENA)
        t_cpu = time.perf_counter() - t0
        for variant in args.variant:
            t0 = time.perf_counter()
            got, stats = distance_transform_cuda(grid, GOAL, n, arena_size=ARENA, check_every=1,
                                                 return_stats=True, variant=variant)
            t_gpu = time.perf_counter() - t0

            same_inf = np.array_equal(np.isinf(ref), np.isinf(got))
            finite = np.isfinite(ref)
            max_err = float(np.max(np.abs(ref[finite] - got[finite]))) if finite.any() else 0.0
            exact = bool(np.array_equal(ref, got))
            good = same_inf and max_err < 1e-9
            ok &= good
            print(f"N={n} [{variant}]: {'PASS' if good else 'FAIL'}  rounds={stats['rounds']}  "
                  f"max|cpu-gpu|={max_err:.2e}  bitwise identical={exact}  "
                  f"same unreachable/wall cells={same_inf}  (cpu {t_cpu:.2f}s, gpu/sim {t_gpu:.2f}s)")
    sys.exit(0 if ok else 1)
