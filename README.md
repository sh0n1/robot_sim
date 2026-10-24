# robot_sim

Multi-robot path-planning simulation (pure pursuit on a grid). The distance transform can be computed sequentially (original) or on an NVIDIA GPU with CUDA (`--cuda`).

## 1. Set up (once)

Create a conda environment from `environment.yml`. The environment is called `simthing` (the `name:` line in that file).

### Windows

1. Open **Anaconda Prompt** from the Start menu.
2. Go to the project folder (the one containing `main.py`):
   ```
   cd C:\Users\<your-name>\Documents\robot_sim
   ```
   If the project is on another drive, switch drive as well: `cd /d D:\path\to\robot_sim`
3. Create and activate the environment:
   ```
   conda env create -f environment.yml
   conda activate simthing
   ```

### Linux

1. Open a terminal.
2. Go to the project folder (the one containing `main.py`):
   ```
   cd ~/Documents/robot_sim
   ```
3. Create and activate the environment:
   ```
   conda env create -f environment.yml
   conda activate simthing
   ```

If the terminal says `conda: command not found`, run `~/miniconda3/bin/conda init bash` once (change the path if conda is installed elsewhere, e.g. `~/anaconda3`), then close and reopen the terminal.

From here on, every command is the same on both systems and is run from the project folder with `simthing` activated (the prompt starts with `(simthing)`).

## 2. Run the simulation (sequential, original)

Set the grid resolution by changing the `N` variable near the top of `main.py` (the line `N = 101`). The values used for benchmarking are 101, 251, 501, 1001 and 2001. Larger grids take much longer: the sequential distance transform takes roughly 4 s at N = 501 and about a minute at N = 2001.

Benchmark with visualisation:

```
python main.py --playback-speed=2 --perf-runs=10 --perf
```

Benchmark without visualisation (recommended):

```
python main.py --perf-runs=10 --perf --no-render
```

Rendering with Matplotlib gets pretty messy and takes very long, so I would keep it off.

`--perf` prints the "Average distance transform time", which is the number to compare between versions.

## 3. Run the distance transform on the GPU (CUDA)

Needs an NVIDIA GPU with a recent driver. Tested on Linux laptop with a GeForce GTX 1650; the same conda packages exist for Windows but I have not tested the CUDA version there.

Install Numba and the CUDA packages into the activated environment (once):

```
conda install -c conda-forge numba cuda-nvvm cuda-cudart cuda-version=12.9
```

Use a `cuda-version` no higher than the CUDA version shown at the top right of the `nvidia-smi` output (12.9 is what I used). Then add `--cuda` to the same commands:

```
python main.py --perf-runs=10 --perf --no-render --cuda
```

The first call compiles the GPU kernels (about a second). `main.py` does this in an untimed warm-up, so it is not included in the reported times. An alternative kernel that updates a single array in place can be selected with `--cuda-variant inplace` (the default is `sync`).

## 4. Check the GPU result is identical

```
python test_correctness.py
```

Compares the CUDA distance map with the original at N = 101 and 251 and prints `PASS` and `bitwise identical=True` when they match exactly.

No GPU? The CUDA simulator runs the same code on the CPU (very slow, small grids only, no meaningful timings):

- Linux: `NUMBA_ENABLE_CUDASIM=1 python test_correctness.py --n 51`
- Windows (Anaconda Prompt):
  ```
  set NUMBA_ENABLE_CUDASIM=1
  python test_correctness.py --n 51
  ```

## 5. Benchmark scripts

| Script | What it does | Output |
|---|---|---|
| `python benchmark_cuda.py` | Times the CUDA and sequential versions side by side, with upload / kernel / download split and the maximum error | `results.csv` |

Useful options: `--n 101 251 501` (grid sizes), `--repeats 5`, `--out file.csv`. For `benchmark_cuda.py` also `--cpu-max-n 0` (skip the slow sequential runs) and `--float32` (faster but not bit-identical; for experiments only). Plug in the charger and avoid other heavy programs while benchmarking.
