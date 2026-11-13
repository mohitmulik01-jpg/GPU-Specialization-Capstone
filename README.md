# GPU Specialization Capstone Project

## GPU-Accelerated Image Processing and Edge Detection Using CUDA C++

**Student:** Mohit Mulik  
**Course:** GPU Specialization – CUDA at Scale  
**Hardware:** NVIDIA Tesla T4  
**Development Environment:** Google Colab  
**Language:** CUDA C++

### Project Overview

This project demonstrates GPU-accelerated image processing using CUDA C++. It implements RGB-to-grayscale conversion and Sobel edge detection using parallel GPU kernels.

A serial CPU implementation is included for correctness checking and performance analysis.

### Main Features

- RGB-to-grayscale conversion
- Sobel edge detection
- CUDA parallel processing
- CPU reference implementation
- GPU execution-time measurement
- Multiple image processing
- CSV benchmark generation
- Output image generation

### Project Structure

- `gpu cap/` – CUDA source code, build files, sample inputs and documentation.
- `results/` – Actual GPU execution logs, benchmark CSV and processed images.

### Hardware and Execution

The project was executed using an NVIDIA Tesla T4 GPU through Google Colab with CUDA Toolkit 13.0.

Three sample images were processed, with five timed GPU iterations per image. The GPU results were compared against the CPU reference, with zero validation mismatches reported.

### How to Run

```bash
cd "gpu cap"
make
bash run.sh
```

### Execution Evidence

See the `results/` folder for:
- Benchmark measurements
- Execution logs
- Three processed output images

### Documentation

See `gpu cap/docs/PROJECT_REPORT.md` for the project report and `gpu cap/docs/PRESENTATION_SCRIPT.md` for the demonstration outline.

### References

- NVIDIA CUDA C++ Programming Guide
- NVIDIA CUDA C++ Best Practices Guide
