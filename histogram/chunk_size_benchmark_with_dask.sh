#!/bin/bash
#SBATCH --nodes=2
#SBATCH --job-name=histogram-split
#SBATCH --qos=debug
#SBATCH --exclusive
#SBATCH --time=00:30:00

export PYTHONPATH=$PYTHONPATH:/gpfs/home/bsc25/bsc25865/tad4bj/src

echo ""
echo "********************"
echo "Calling dask-prepare"
echo "********************"
$PWD/../dask-prepare.sh

echo ""
echo "********************"
echo "Dask should be ready"
echo "Ready to start"
echo "********************"

echo " * * * histogram-dask.py * * * "
~/anaconda3/bin/python -u $PWD/chunk_size_benchmark-dask.py

echo ""
echo "********"
echo "Finished"
echo "********"

tad4bj setnow finish_ts
