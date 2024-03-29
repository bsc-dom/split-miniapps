#!/bin/bash -e

  module load gcc/8.1.0
  export COMPSS_PYTHON_VERSION=3-ML
  module load COMPSs/2.10
  module load mkl/2018.1
  module load impi/2018.1
  module load opencv/4.1.2
  #module load DATACLAY/DevelAlex

  # Retrieve script arguments
  num_nodes=${1:-3}
  execution_time=${2:-120}
  tracing=${3:-false}

  # Freeze storage_props into a temporal
  # (allow submission of multiple executions with varying parameters)
  STORAGE_PROPS=`mktemp -p ~`
  cp $(pwd)/cfgfiles/storage_props.cfg "${STORAGE_PROPS}"
  source ${STORAGE_PROPS}

  source execution_values

  TAD4BJ_JSON=`mktemp -p ~`
  cat << EOF > $TAD4BJ_JSON
{
  "dataclay": 0,
  "nodes": ${num_nodes},
  "backends_per_node": ${BACKENDS_PER_NODE},
  "cpus_per_node": ${CPUS_PER_NODE},
  "computing_units": ${COMPUTING_UNITS},
  "use_split": 0,
  "randomize_pairings": ${RANDOMIZE_PAIRINGS},
  "blocksize": ${BLOCKSIZE},
  "col_blocks": ${COL_BLOCKS},
  "row_blocks": ${ROW_BLOCKS}
}
EOF

  # Define script variables
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR=${SCRIPT_DIR}/
  APP_CLASSPATH=${SCRIPT_DIR}/
  APP_PYTHONPATH=${SCRIPT_DIR}/
  # $USER is "built-in", $GROUP is not
  GROUP=$(id -g -n $USER)
  #WORKER_WORKING_DIR=/gpfs/scratch/$GROUP/$USER
  WORKER_WORKING_DIR=local_disk

  # Define application variables
  graph=$tracing
  log_level="off"
  qos_flag=${QOS_FLAG:---qos=debug}
  workers_flag=""
  constraints=""

  WORKER_IN_MASTER=0

  # Those are evaluated at submit time, not at start time...
  COMPSS_VERSION=`ml whatis COMPSs 2>&1 >/dev/null | awk '{print $1 ; exit}'`
  DATACLAY_VERSION=`ml whatis DATACLAY 2>&1 >/dev/null | awk '{print $1 ; exit}'`

  # Note that this is needed also in master, so it must be put here (not in --pythonpath)
  export PYTHONPATH=$PYTHONPATH:~/dislib-trunk

  export ComputingUnits=${COMPUTING_UNITS}

  # Enqueue job
  enqueue_compss \
    --job_name=svd-split \
    --exec_time="${execution_time}" \
    --num_nodes="${num_nodes}" \
    \
    --cpus_per_node="${CPUS_PER_NODE}" \
    --worker_in_master_cpus="${WORKER_IN_MASTER}" \
    --scheduler=es.bsc.compss.scheduler.fifodatalocation.FIFODataLocationScheduler \
    \
    "${workers_flag}" \
    \
    --worker_working_dir=$WORKER_WORKING_DIR \
    \
    --constraints=${constraints} \
    --tracing="${tracing}" \
    --graph="${graph}" \
    --summary \
    --log_level="${log_level}" \
    "${qos_flag}" \
    \
    --classpath=${DATACLAY_JAR} \
    --pythonpath=${APP_PYTHONPATH}:${PYTHONPATH} \
    --prolog="tad4bj,setnow,start_ts" \
    --prolog="tad4bj,setdict,$TAD4BJ_JSON" \
    --epilog="tad4bj,setnow,finish_ts" \
    \
    ${extra_tracing_flags} \
    \
    --lang=python \
    \
    "$(pwd)/svd_nodc.py"
