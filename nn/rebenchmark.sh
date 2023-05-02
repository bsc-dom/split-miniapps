#!/bin/bash

#SBATCH --job-name=nn_benchmark
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --time=00:20:00

ml gcc/8.1.0
ml impi/2018.1
ml mkl/2018.1
ml opencv/4.1.2
ml python/3.6.4_ML

python rebenchmark.py