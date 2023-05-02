#!/bin/bash

#SBATCH --job-name=histogram_benchmark
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --exclusive
#SBATCH --time=02:00:00

numactl --cpunodebind=0 --membind=0 -- python -u chunk_size_benchmark.py
