import math
import time

import numpy as np
from numba import cuda

SQRT2 = math.sqrt(2.0)
DEFAULT_BLOCK = (16, 16)  # 256 threads per block


@cuda.jit
def relax_round(wall, d_in, d_out, changed, n):
    """One synchronous relaxation round. wall: 1 = obstacle, 0 = free."""
    x, y = cuda.grid(2)
    if x >= n or y >= n:
        return
    if wall[y, x] != 0:
        d_out[y, x] = math.inf  # walls stay at infinity
        return

    best = d_in[y, x]
    # 8-neighbourhood, same rule as the sequential code (including diagonals).
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            if wall[ny, nx] != 0:
                continue
            step = SQRT2 if (dx != 0 and dy != 0) else 1.0
            cand = d_in[ny, nx] + step
            if cand < best:
                best = cand
    d_out[y, x] = best
    # Compare the value as STORED (after any float32 rounding) with the old one.  Comparing the
    # unrounded `best` could stay "smaller" forever for float32 and the loop would never converge.
    if d_out[y, x] < d_in[y, x]:
        changed[0] = 1  # benign race: every writer writes the same value (1)


@cuda.jit
def relax_inplace(wall, d, changed, n):
    """One in-place (asynchronous) relaxation round on a single buffer."""
    x, y = cuda.grid(2)
    if x >= n or y >= n:
        return
    if wall[y, x] != 0:
        return  # walls were initialised to infinity and never change

    old = d[y, x]
    best = old
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= n or ny < 0 or ny >= n:
                continue
            if wall[ny, nx] != 0:
                continue
            step = SQRT2 if (dx != 0 and dy != 0) else 1.0
            cand = d[ny, nx] + step
            if cand < best:
                best = cand
    if best < old:
        d[y, x] = best
        if d[y, x] < old:  # as stored: float32 rounding can store `old` again
            changed[0] = 1


def distance_transform_cuda(occupancy, goal, N, arena_size=2.0,
                            check_every=16, dtype=np.float64, return_stats=False,
                            block=DEFAULT_BLOCK, variant="sync", max_rounds=None):
    """Drop-in replacement for controllers.distance_transform (same arguments).

    check_every: rounds to run between convergence checks.  Each check copies
        one int back to the host (a sync point), so a larger value means fewer
        syncs but up to check_every-1 wasted rounds at the end.
    dtype: np.float64 matches the CPU code exactly; np.float32 is faster on
        consumer GPUs (FP64 runs at 1/32 rate on GeForce) but is only approximate
        (rounding error accumulates along long paths; measure it, don't assume).
    block: threads per block as (x, y).
    variant: "sync" (double buffered) or "inplace" (single buffer).
    max_rounds: safety cap (default 20*N + 200; real runs need about 0.8*N).
        Raises RuntimeError instead of looping forever if convergence is never reached.
    """
    if max_rounds is None:
        max_rounds = 20 * N + 200
    if variant not in ("sync", "inplace"):
        raise ValueError("variant must be 'sync' or 'inplace'")
    cell_size = arena_size / (N - 1)
    goal_x = int(round(goal[0] / cell_size))
    goal_y = int(round(goal[1] / cell_size))

    t0 = time.perf_counter()
    wall_h = np.ascontiguousarray(occupancy != 0, dtype=np.uint8)
    dist_h = np.full((N, N), np.inf, dtype=dtype)
    dist_h[goal_y, goal_x] = 0

    wall = cuda.to_device(wall_h)
    d_a = cuda.to_device(dist_h)
    d_b = cuda.to_device(dist_h) if variant == "sync" else None
    changed = cuda.to_device(np.zeros(1, dtype=np.int32))
    cuda.synchronize()
    t_upload = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = ((N + block[0] - 1) // block[0], (N + block[1] - 1) // block[1])
    rounds = 0
    while True:
        changed.copy_to_device(np.zeros(1, dtype=np.int32))
        for _ in range(check_every):
            if variant == "sync":
                relax_round[grid, block](wall, d_a, d_b, changed, N)
                d_a, d_b = d_b, d_a  # swap roles: last output becomes next input
            else:
                relax_inplace[grid, block](wall, d_a, changed, N)
            rounds += 1
        if changed.copy_to_host()[0] == 0:  # nothing changed in ANY round of this batch
            break
        if rounds >= max_rounds:
            raise RuntimeError(f"no convergence after {rounds} rounds (variant={variant}, N={N}, "
                               f"dtype={np.dtype(dtype).name}, block={block})")
    cuda.synchronize()
    t_kernels = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = d_a.copy_to_host()
    t_download = time.perf_counter() - t0
    if return_stats:
        return result, {"rounds": rounds, "upload_s": t_upload,
                        "kernels_s": t_kernels, "download_s": t_download}
    return result