#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
              << ": " << cudaGetErrorString(err) << "\n"; \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;
};

static void SkipComments(std::istream& in) {
  while (in >> std::ws && in.peek() == '#') {
    std::string line;
    std::getline(in, line);
  }
}

static Image ReadPpm(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open input: " + path);
  std::string magic;
  in >> magic;
  if (magic != "P6") {
    throw std::runtime_error("Only binary PPM (P6) is supported: " + path);
  }
  SkipComments(in);
  Image image;
  int max_value = 0;
  in >> image.width;
  SkipComments(in);
  in >> image.height;
  SkipComments(in);
  in >> max_value;
  in.get();  // Consume the single whitespace character before pixel bytes.
  if (image.width <= 0 || image.height <= 0 || max_value != 255) {
    throw std::runtime_error("Invalid PPM dimensions or max value in " + path);
  }
  const size_t bytes = static_cast<size_t>(image.width) * image.height * 3;
  image.rgb.resize(bytes);
  in.read(reinterpret_cast<char*>(image.rgb.data()),
          static_cast<std::streamsize>(bytes));
  if (!in) throw std::runtime_error("Truncated PPM pixel data: " + path);
  return image;
}

static void WritePpm(const std::string& path, int width, int height,
                     const std::vector<uint8_t>& gray) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot write output: " + path);
  out << "P6\n" << width << " " << height << "\n255\n";
  for (uint8_t value : gray) {
    const char pixel[3] = {static_cast<char>(value),
                           static_cast<char>(value),
                           static_cast<char>(value)};
    out.write(pixel, 3);
  }
}

__global__ void RgbToGrayKernel(const uint8_t* rgb, uint8_t* gray,
                                int pixel_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) return;
  const int offset = index * 3;
  const float value = 0.299f * rgb[offset] +
                      0.587f * rgb[offset + 1] +
                      0.114f * rgb[offset + 2];
  gray[index] = static_cast<uint8_t>(value + 0.5f);
}

__global__ void SobelKernel(const uint8_t* gray, uint8_t* edges,
                            int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int index = y * width + x;
  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    edges[index] = 0;
    return;
  }

  const int p = index;
  const int gx = -gray[p - width - 1] + gray[p - width + 1]
                 - 2 * gray[p - 1] + 2 * gray[p + 1]
                 - gray[p + width - 1] + gray[p + width + 1];
  const int gy = -gray[p - width - 1] - 2 * gray[p - width]
                 - gray[p - width + 1] + gray[p + width - 1]
                 + 2 * gray[p + width] + gray[p + width + 1];
  const float magnitude = sqrtf(static_cast<float>(gx * gx + gy * gy));
  edges[index] = static_cast<uint8_t>(fminf(255.0f, magnitude));
}

static std::vector<uint8_t> CpuPipeline(const Image& image) {
  const int width = image.width;
  const int height = image.height;
  const int count = width * height;
  std::vector<uint8_t> gray(count), edges(count, 0);
  for (int i = 0; i < count; ++i) {
    const int p = i * 3;
    gray[i] = static_cast<uint8_t>(
        0.299f * image.rgb[p] + 0.587f * image.rgb[p + 1] +
        0.114f * image.rgb[p + 2] + 0.5f);
  }
  for (int y = 1; y < height - 1; ++y) {
    for (int x = 1; x < width - 1; ++x) {
      const int p = y * width + x;
      const int gx = -gray[p - width - 1] + gray[p - width + 1]
                     - 2 * gray[p - 1] + 2 * gray[p + 1]
                     - gray[p + width - 1] + gray[p + width + 1];
      const int gy = -gray[p - width - 1] - 2 * gray[p - width]
                     - gray[p - width + 1] + gray[p + width - 1]
                     + 2 * gray[p + width] + gray[p + width + 1];
      edges[p] = static_cast<uint8_t>(
          std::min(255.0f, std::sqrt(static_cast<float>(gx * gx + gy * gy))));
    }
  }
  return edges;
}

static double Milliseconds(
    const std::chrono::steady_clock::time_point& start,
    const std::chrono::steady_clock::time_point& finish) {
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

int main(int argc, char** argv) {
  std::string input_path, output_path = "edge_output.ppm";
  std::string csv_path = "results/benchmark.csv";
  int threads = 256;
  int repeats = 5;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value after " + arg);
      return argv[++i];
    };
    if (arg == "--input") input_path = next();
    else if (arg == "--output") output_path = next();
    else if (arg == "--csv") csv_path = next();
    else if (arg == "--threads") threads = std::stoi(next());
    else if (arg == "--repeat") repeats = std::stoi(next());
    else if (arg == "--help") {
      std::cout << "Usage: ./gpu_image_capstone --input FILE.ppm "
                   "[--output FILE.ppm] [--csv FILE.csv] "
                   "[--threads 128|256|512] [--repeat N]\n";
      return 0;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (input_path.empty() || repeats < 1 || threads < 1 || threads > 1024) {
    std::cerr << "Input is required; repeat must be >=1 and threads 1..1024.\n";
    return EXIT_FAILURE;
  }

  try {
    const Image image = ReadPpm(input_path);
    const int width = image.width, height = image.height;
    const int count = width * height;
    const size_t rgb_bytes = image.rgb.size();
    const size_t gray_bytes = static_cast<size_t>(count);

    const auto cpu_start = std::chrono::steady_clock::now();
    std::vector<uint8_t> cpu_edges = CpuPipeline(image);
    const auto cpu_finish = std::chrono::steady_clock::now();
    const double cpu_ms = Milliseconds(cpu_start, cpu_finish);

    uint8_t *d_rgb = nullptr, *d_gray = nullptr, *d_edges = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
    CUDA_CHECK(cudaMalloc(&d_edges, gray_bytes));
    CUDA_CHECK(cudaMemcpy(d_rgb, image.rgb.data(), rgb_bytes,
                          cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);
    (void)threads;  // Retained as a CLI tuning parameter for reproducible runs.
    const int linear_blocks = (count + threads - 1) / threads;

    // Warm-up avoids including first-use runtime initialization in the sample.
    RgbToGrayKernel<<<linear_blocks, threads>>>(d_rgb, d_gray, count);
    SobelKernel<<<grid, block>>>(d_gray, d_edges, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    for (int iteration = 0; iteration < repeats; ++iteration) {
      CUDA_CHECK(cudaEventRecord(start));
      RgbToGrayKernel<<<linear_blocks, threads>>>(d_rgb, d_gray, count);
      SobelKernel<<<grid, block>>>(d_gray, d_edges, width, height);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
      samples.push_back(elapsed);
    }

    std::vector<uint8_t> gpu_edges(count);
    CUDA_CHECK(cudaMemcpy(gpu_edges.data(), d_edges, gray_bytes,
                          cudaMemcpyDeviceToHost));
    int mismatches = 0;
    int max_error = 0;
    for (int i = 0; i < count; ++i) {
      const int error = std::abs(static_cast<int>(gpu_edges[i]) -
                                 static_cast<int>(cpu_edges[i]));
      if (error > 1) ++mismatches;
      max_error = std::max(max_error, error);
    }
    WritePpm(output_path, width, height, gpu_edges);

    std::ofstream csv(csv_path, std::ios::app);
    if (!csv) throw std::runtime_error("Cannot write CSV: " + csv_path);
    if (csv.tellp() == 0) {
      csv << "input,width,height,pixels,cpu_ms,gpu_kernel_ms,repeat,"
             "threads_per_block,mismatches,max_abs_error\n";
    }
    for (size_t i = 0; i < samples.size(); ++i) {
      csv << input_path << "," << width << "," << height << "," << count
          << "," << std::fixed << std::setprecision(4) << cpu_ms << ","
          << samples[i] << "," << (i + 1) << "," << threads << ","
          << mismatches << "," << max_error << "\n";
    }

    cudaDeviceProp prop{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << "\n"
              << "Input: " << input_path << " (" << width << "x" << height
              << ")\nCPU pipeline: " << cpu_ms << " ms\n"
              << "GPU kernel samples: ";
    for (float value : samples) std::cout << value << " ms ";
    std::cout << "\nValidation mismatches (>1 intensity): " << mismatches
              << ", maximum absolute error: " << max_error << "\n"
              << "Output: " << output_path << "\nCSV: " << csv_path << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_edges));
    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
