"""Print Python, scientific-package, NVIDIA, and CUDA environment details."""

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def conda_packages():
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda is None:
        return {}

    output = run_command([conda, "list", "--json"])
    if output is None:
        return {}

    try:
        packages = json.loads(output)
    except json.JSONDecodeError:
        return {}

    cuda_names = ("cuda", "cudatoolkit", "cudnn", "nvvm")
    return {
        package["name"]: package.get("version", "unknown")
        for package in packages
        if package.get("name", "").startswith(cuda_names)
    }


def print_cuda_status():
    try:
        from numba import cuda
    except ImportError as error:
        print(f"Numba CUDA: unavailable ({error})")
        return

    try:
        available = cuda.is_available()
    except Exception as error:
        print(f"Numba CUDA: error checking availability ({error})")
        return

    print(f"Numba CUDA available: {available}")
    if available:
        try:
            print(f"CUDA device: {cuda.get_current_device().name.decode()}")
        except Exception as error:
            print(f"CUDA device: unavailable ({error})")


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print(f"NumPy: {package_version('numpy')}")
    print(f"Numba: {package_version('numba')}")

    driver = run_command([
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader,nounits",
    ])
    print(f"NVIDIA driver/GPU: {driver or 'unavailable'}")

    packages = conda_packages()
    if packages:
        print("CUDA Conda packages:")
        for name, version in sorted(packages.items()):
            print(f"  {name}: {version}")
    else:
        print("CUDA Conda packages: unavailable")

    print_cuda_status()


if __name__ == "__main__":
    main()