"""CUDA (Numba) distance transform: synchronous, round-based relaxation.

Sequential version (controllers.distance_transform): one FIFO queue, one cell
relaxed at a time.  Inherently serial, but only does O(N^2) work.

This version: one GPU thread per grid cell.  Every round, every free cell looks
at its 8 neighbours in the OLD distance buffer and writes
    new[y, x] = min(old[y, x], min over neighbours(old[neighbour] + step cost))
into a SEPARATE buffer (double buffering, so no thread ever reads a value
another thread is writing in the same round -> no data race, deterministic).
Rounds repeat until a whole round changes nothing (fixed point = the same
shortest-path distances the queue version computes).

Trade-off to discuss in the report: this does more total work (about
N^2 cells x ~O(N) rounds) than the queue version, but every round is fully
parallel, so a GPU can still win by a large factor.
"""
import math
import time

import numpy as np
from numba import cuda

SQRT2 = math.sqrt(2.0)
BLOCK = (16, 16)  # 256 threads per block


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
    if best < d_in[y, x]:
        changed[0] = 1  # benign race: every writer writes the same value (1)


def distance_transform_cuda(occupancy, goal, N, arena_size=2.0,
                            check_every=16, dtype=np.float64, return_stats=False):
    """Drop-in replacement for controllers.distance_transform (same arguments).

    check_every: how many rounds to run between convergence checks.  Each check
        copies one int back to the host (a sync point), so a larger value means
        fewer syncs but up to check_every-1 wasted rounds at the end.
    dtype: np.float64 matches the CPU code; np.float32 is faster on consumer
        GPUs (1/32-rate FP64 on GeForce) but only agrees to ~1e-4.
    """
    cell_size = arena_size / (N - 1)
    goal_x = int(round(goal[0] / cell_size))
    goal_y = int(round(goal[1] / cell_size))

    t0 = time.perf_counter()
    wall_h = np.ascontiguousarray(occupancy != 0, dtype=np.uint8)
    dist_h = np.full((N, N), np.inf, dtype=dtype)
    dist_h[goal_y, goal_x] = 0

    wall = cuda.to_device(wall_h)
    d_a = cuda.to_device(dist_h)
    d_b = cuda.to_device(dist_h)
    changed = cuda.to_device(np.zeros(1, dtype=np.int32))
    cuda.synchronize()
    t_upload = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = ((N + BLOCK[0] - 1) // BLOCK[0], (N + BLOCK[1] - 1) // BLOCK[1])
    rounds = 0
    while True:
        changed.copy_to_device(np.zeros(1, dtype=np.int32))
        for _ in range(check_every):
            relax_round[grid, BLOCK](wall, d_a, d_b, changed, N)
            d_a, d_b = d_b, d_a  # swap roles: last output becomes next input
            rounds += 1
        if changed.copy_to_host()[0] == 0:  # nothing changed in ANY round of this batch
            break
    cuda.synchronize()
    t_kernels = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = d_a.copy_to_host()
    t_download = time.perf_counter() - t0
    if return_stats:
        return result, {"rounds": rounds, "upload_s": t_upload,
                        "kernels_s": t_kernels, "download_s": t_download}
    return result
