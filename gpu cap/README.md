# GPU-Accelerated Image Edge Detection (CUDA C++)

## Project overview
This capstone implements an RGB-to-grayscale conversion followed by Sobel edge detection. It compares a serial CPU reference implementation with two CUDA kernels. The GPU version uses a two-dimensional grid of thread blocks for the image operation and explicit device memory transfers. The project is intentionally dependency-light: input and output images use the binary PPM (P6) format, so OpenCV is not required.

## Why this project
Image filtering is a useful example of data-parallel computation: each output pixel can be computed independently after its neighborhood is available. The work demonstrates CUDA kernel launches, device memory allocation, host-to-device/device-to-host transfers, event-based timing, correctness checking, and repeatable experiments.

## Requirements
- Linux environment (native Linux, WSL2 with CUDA support, or a CUDA-enabled course lab)
- NVIDIA GPU supported by the installed driver
- NVIDIA CUDA Toolkit (`nvcc`)
- GNU Make and a C++17-capable host compiler
- Python 3 only to generate the included deterministic sample images (not needed to run the CUDA executable)

Check the environment:
```bash
nvidia-smi
nvcc --version
```
The CUDA Programming Guide describes compiling CUDA C++ with `nvcc`, launching `__global__` kernels, and managing device memory. See the official references at the end of this README.

## Build
```bash
make
```

## Run one image
```bash
mkdir -p results
./gpu_image_capstone --input data/sample_512.ppm \
  --output results/sample_edges.ppm \
  --csv results/benchmark.csv --threads 256 --repeat 5
```
Arguments:
- `--input`: required P6 PPM image
- `--output`: output edge image (P6, grayscale replicated across RGB channels)
- `--csv`: append benchmark records to this CSV
- `--threads`: recorded tuning label (the current 2D Sobel launch uses 16x16 threads; the linear grayscale launch uses the requested `--threads` value)
- `--repeat`: number of timed GPU kernel iterations (must be at least 1)
- `--help`: print usage

To regenerate the deterministic sample inputs, run `python3 tools_generate_samples.py`.

## Run the supplied multi-image experiment
```bash
bash run.sh
```
The script rebuilds the executable, runs every `data/*.ppm` input five times, writes edge images to `results/`, and appends per-iteration measurements to `results/benchmark.csv`.

## Algorithms
1. **RGB to grayscale:** `Y = 0.299R + 0.587G + 0.114B`.
2. **Sobel operator:** a 3x3 horizontal and vertical gradient stencil is applied to each interior pixel. The gradient magnitude is `sqrt(Gx*Gx + Gy*Gy)`, clipped to 255. The outermost border is set to zero.
3. **CPU reference:** performs the same two stages serially and provides a correctness reference. The GPU and CPU outputs are compared; the program fails if any pixel differs by more than one intensity level.

## Timing methodology and limitations
The CPU time covers its full grayscale-plus-Sobel pipeline. CUDA event time covers the two GPU kernels only; it excludes host/device transfers and allocation. These are different timing scopes and must not be presented as an end-to-end speedup comparison. For a fair end-to-end comparison, add transfer and allocation timing or report the kernel-only and transfer-inclusive measurements separately. The GPU kernel is warmed up before timed iterations. Results depend on GPU model, driver, toolkit, image dimensions, clock state, and system load.

## Evidence to commit
After running on a real CUDA-capable GPU, commit:
- `results/benchmark.csv`
- `results/*_edges.ppm`
- terminal output or a captured log from `bash run.sh`
- screenshots showing `nvidia-smi`, successful execution, and output images

Do not invent or manually alter benchmark measurements. If no CUDA GPU is available, state that execution has not yet been validated on GPU.

## Suggested GitHub publication
```bash
git init
git add .
git commit -m "Add CUDA image processing capstone"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/gpu-image-capstone.git
git push -u origin main
```
Replace the URL with your own public repository URL.

## References
- NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/): CUDA C++ kernels, launch configuration, runtime and device memory.
- NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/): performance considerations including host/device transfer costs.
