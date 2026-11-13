#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
make clean
make
: > results/benchmark.csv
for input in data/*.ppm; do
  [[ -e "$input" ]] || { echo "No PPM inputs found in data/"; exit 1; }
  name="$(basename "$input" .ppm)"
  ./gpu_image_capstone --input "$input" \
    --output "results/${name}_edges.ppm" \
    --csv results/benchmark.csv --threads 256 --repeat 5
done
echo "Completed. Review results/benchmark.csv and results/*_edges.ppm."
