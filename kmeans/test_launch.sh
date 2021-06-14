#!/bin/bash -e

  module load COMPSs/2.8.1

  # Enqueue job
  enqueue_compss \
    --job_name=test \
    --exec_time=10 \
    --num_nodes=3 \
    \
    --cpus_per_node=48 \
    --worker_in_master_cpus=0 \
    --scheduler=es.bsc.compss.scheduler.fifodatalocation.FIFODataLocationScheduler \
    --worker_working_dir=local_disk \
    \
    --qos=debug \
    \
    --lang=python \
    \
    "$(pwd)/test.py"
