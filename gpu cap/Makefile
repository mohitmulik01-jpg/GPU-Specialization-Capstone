NVCC ?= nvcc
TARGET := gpu_image_capstone
SRC := src/main.cu
NVCCFLAGS ?= -O2 -std=c++17 -lineinfo

.PHONY: all clean run
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	bash run.sh

clean:
	rm -f $(TARGET) edge_output*.ppm results/benchmark.csv
