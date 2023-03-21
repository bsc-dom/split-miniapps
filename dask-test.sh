#!/bin/bash
#SBATCH --nodes=3
#SBATCH --time=1:00:00
#SBATCH --job-name=TestingDask
#SBATCH --qos=debug

echo ""
echo "********************"
echo "Calling dask-prepare"
echo "********************"
./dask-prepare.sh

echo ""
echo "********************"
echo "Dask should be ready"
echo "********************"

sleep 3600
