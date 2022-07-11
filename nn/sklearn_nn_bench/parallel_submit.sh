#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --exclusive
#SBATCH --job-name="nn_bench_parallel"

  module load gcc/8.1.0
  module load python/3.6.4_ML
  module load mkl/2018.1
  module load impi/2018.1
  module load opencv/4.1.2

echo "We are in $PWD. Ready to start applications"

echo "Starting fit_parallel on socket 0"
numactl -N 0 -m 0 -- python -u fit_parallel.py > fit.out &

echo "Starting kneighbors_parallel on socket 1"
numactl -N 1 -m 1 -- python -u kneighbors_parallel.py > kneighbors.out &

echo "Waiting..."
wait
echo "Done!"
