#!/bin/bash

GROUP=$(id -g -n $USER)
LOG_OUTPUT_DIR=/gpfs/scratch/$GROUP/$USER

node_list=$SLURM_JOB_NODELIST
nodes=($(scontrol show hostname $node_list))

scheduler=${nodes[0]}
workers=(${nodes[@]:1})
scheduler_file=~/dask-scheduler-$SLURM_JOB_ID.json


cat <<EOF >~/.config/dask/distributed.yaml
distributed:
  worker:
    memory:
      target: false  # don't spill to disk
      spill: false  # don't spill to disk
      pause: 0.90  # pause execution at 80% memory use
      terminate: 0.95  # restart the worker at 95% use
EOF

echo "Starting scheduler on $scheduler_file"
ssh $scheduler "nohup ./anaconda3/bin/dask-scheduler --scheduler-file $scheduler_file --interface ib0 >$LOG_OUTPUT_DIR/dask-scheduler-$SLURM_JOB_ID.out 2>$LOG_OUTPUT_DIR/dask-scheduler-$SLURM_JOB_ID.err &"

for node in "${workers[@]}" ; do
    echo "Starting workers (one per socket) on $node"
    ssh $node "nohup numactl --cpunodebind=0 --membind=0 -- ./anaconda3/bin/dask-worker --nworkers 1 --nthreads 24 --scheduler-file $scheduler_file --interface ib0 >$LOG_OUTPUT_DIR/dask-worker-0-$SLURM_JOB_ID.out 2>$LOG_OUTPUT_DIR/dask-worker-0-$SLURM_JOB_ID.err &"
    ssh $node "nohup numactl --cpunodebind=1 --membind=1 -- ./anaconda3/bin/dask-worker --nworkers 1 --nthreads 24 --scheduler-file $scheduler_file --interface ib0 >$LOG_OUTPUT_DIR/dask-worker-1-$SLURM_JOB_ID.out 2>$LOG_OUTPUT_DIR/dask-worker-1-$SLURM_JOB_ID.err &"
done
