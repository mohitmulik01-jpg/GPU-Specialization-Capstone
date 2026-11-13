# GPU Specialization Capstone Project Report

**Project:** GPU-Accelerated Image Edge Detection Using CUDA C++  
**Student:** [Enter your name]  
**Course:** GPU Specialization / CUDA at Scale  
**Date:** [Enter submission date]  
**Repository:** [Paste your public repository URL]

## Abstract
This project implements a small image-processing pipeline using CUDA C++. The pipeline converts RGB pixels to grayscale and applies a Sobel edge detector. A serial CPU implementation is included as a reference. The GPU implementation distributes pixel work across CUDA threads and uses explicit device memory management. The program processes multiple PPM images, records repeated kernel timings, saves output images, and checks GPU results against the CPU reference.

## 1. Problem and motivation
Image operations often apply the same calculation to many pixels. This makes image processing a suitable example for data-parallel execution. The project investigates how a simple stencil-based image operation can be expressed as CUDA kernels and what measurements are needed to evaluate it responsibly.

## 2. Design and implementation
The application accepts command-line arguments for input image, output image, CSV destination, repeat count, and a thread-configuration label. It reads binary PPM (P6) files to avoid external image-library dependencies.

### 2.1 Grayscale conversion
For each RGB pixel, the application computes:
`Y = 0.299R + 0.587G + 0.114B`.
The CUDA kernel assigns one thread to each pixel.

### 2.2 Sobel edge detection
The second kernel reads a 3x3 neighborhood and computes horizontal and vertical gradients using the Sobel masks. The magnitude is clipped to the 8-bit range. Border pixels are set to zero. A two-dimensional launch maps CUDA thread coordinates to image coordinates.

### 2.3 CPU reference and validation
The CPU implementation performs the same two stages serially. The program compares CPU and GPU edge images and reports the mismatch count and maximum absolute pixel error. The program returns a failure status if any pixel differs by more than one intensity value.

## 3. Experimental method
The supplied script runs all PPM inputs in `data/`. Each image is processed for five timed GPU iterations after a warm-up. CUDA events measure the two GPU kernels. CPU wall-clock timing measures the complete CPU pipeline.

**Important measurement distinction:** GPU kernel timing excludes allocation and host/device copies, while CPU timing includes its full pipeline. Therefore these figures are not a direct end-to-end speedup comparison. A future experiment should separately measure transfer-inclusive GPU time and compare equivalent timing scopes.

## 4. Results
Populate this section only after running the program on the target CUDA system.

| Input image | Dimensions | CPU pipeline (ms) | GPU kernel median (ms) | Mismatches | GPU model |
|---|---:|---:|---:|---:|---|
| [sample] | [W x H] | [from CSV] | [from CSV] | [from log] | [from nvidia-smi] |

Attach the generated `results/benchmark.csv`, edge images, terminal log, and screenshots. Do not insert estimated or fabricated values.

## 5. Challenges and lessons learned
Discuss the actual issues encountered during compilation and execution, including CUDA Toolkit compatibility, input format handling, kernel launch dimensions, boundary conditions, synchronization, and measurement scope. Replace this guidance with your own observations after running the project.

## 6. Limitations and future work
- The current input reader supports binary PPM (P6) only.
- The current kernels use straightforward global-memory accesses; shared-memory tiling could reduce redundant neighborhood loads.
- The `--threads` argument controls the 1D grayscale kernel block size; the 2D Sobel launch remains fixed at 16x16. Compare settings only on the same GPU and workload.
- Future work could add Gaussian blur, tiled Sobel, pinned-memory transfers, streams, and tests on multiple image sizes and GPU architectures.

## 7. Reproduction instructions
See `README.md`. In summary: install a compatible NVIDIA driver and CUDA Toolkit, run `make`, then `bash run.sh`. Preserve the generated CSV, images, and console log in the repository.

## 8. References
1. NVIDIA, *CUDA C++ Programming Guide*, https://docs.nvidia.com/cuda/cuda-programming-guide/
2. NVIDIA, *CUDA C++ Best Practices Guide*, https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
