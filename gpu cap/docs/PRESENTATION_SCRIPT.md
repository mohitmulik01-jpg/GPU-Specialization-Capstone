# 5–7 Minute Project Demonstration Script

**Before recording:** run `nvidia-smi`, `nvcc --version`, and `bash run.sh` on a CUDA-capable machine. Keep the actual terminal output and generated results visible. Do not claim GPU execution if it has not occurred.

## 0:00–0:45 — Introduction
“Hello. My capstone is a CUDA C++ image-processing application. It converts RGB images to grayscale and applies Sobel edge detection. I chose this because each pixel uses a similar calculation, making it a clear example of data-parallel GPU work.”

## 0:45–1:30 — Project structure
“Here are the main files: `src/main.cu` contains the CPU reference and CUDA kernels, `Makefile` builds the executable, `run.sh` processes the sample image set, and `results/` stores output images and benchmark records.”

## 1:30–2:30 — Explain the kernels
“The first kernel converts each RGB pixel into a grayscale intensity. The second kernel uses a 3-by-3 Sobel neighborhood to calculate horizontal and vertical gradients. CUDA thread and block coordinates map to image coordinates. Boundary checks prevent threads outside the image from reading invalid pixels.”

## 2:30–3:30 — Run the application
“Now I will run the supplied script. It builds the program and processes each PPM image five times. The program reports the GPU model, CPU reference time, GPU kernel samples, and correctness comparison.”

Show the actual command and output.

## 3:30–4:30 — Show evidence
“Here are the edge images and the CSV file. Each CSV row represents one timed GPU iteration. The mismatch count checks that the GPU output agrees with the CPU reference within the selected one-level tolerance.”

Show the generated images and open the CSV.

## 4:30–5:45 — Discuss measurements honestly
“The CPU measurement includes the complete CPU pipeline. The CUDA event measurement covers only the two kernels, not memory transfers or allocation. These timing scopes are different, so I do not interpret their ratio as an end-to-end speedup. A more complete performance study would time transfers and allocation separately and optimize memory reuse.”

## 5:45–6:30 — Challenges and next steps
“The main engineering considerations were image boundary handling, correct launch geometry, memory transfers, synchronization, and reproducible timing. Next I would test shared-memory tiling, additional image sizes, and transfer-inclusive timings.”

## 6:30–7:00 — Conclusion
“This project demonstrates a complete CUDA image-processing workflow: input handling, CPU reference, GPU kernels, output generation, repeated timing, and correctness validation. Thank you.”
