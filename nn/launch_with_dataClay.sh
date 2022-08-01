#!/bin/bash -e

  module load gcc/8.1.0
  export COMPSS_PYTHON_VERSION=3-ML
  module load COMPSs/2.10
  module load mkl/2018.1
  module load impi/2018.1
  module load opencv/4.1.2
  module load DATACLAY/DevelAlex

  export COMPSS_LOAD_SOURCE=false

  # Retrieve script arguments
  num_nodes=${1:-3}
  execution_time=${2:-60}
  tracing=${3:-true}

  if [[ $tracing == "true" ]]  ; then
    TRACING_FLAG=1
  else
    TRACING_FLAG=0
  fi

  # Freeze storage_props into a temporal
  # (allow submission of multiple executions with varying parameters)
  STORAGE_PROPS=`mktemp -p ~`
  cp $(pwd)/cfgfiles/storage_props.cfg "${STORAGE_PROPS}"
  source ${STORAGE_PROPS}

  export DATACLAYSESSIONCONFIG=$STORAGE_PROPS

  source execution_values

  TAD4BJ_JSON=`mktemp -p ~`
  cat << EOF > $TAD4BJ_JSON
{
  "dataclay": 1,
  "tracing": ${TRACING_FLAG},
  "nodes": ${num_nodes},
  "num_threads": ${OPENBLAS_NUM_THREADS:-48},
  "backends_per_node": ${BACKENDS_PER_NODE},
  "cpus_per_node": ${CPUS_PER_NODE},
  "compss_scheduler": "${COMPSS_SCHEDULER}",
  "compss_working_dir": "${COMPSS_WORKING_DIR}",
  "computing_units": ${COMPUTING_UNITS},
  "use_split": ${USE_SPLIT},
  "copy_fit_struct": ${COPY_FIT_STRUCT},
  "points_per_block": ${POINTS_PER_BLOCK},
  "n_blocks_fit": ${N_BLOCKS_FIT},
  "n_blocks_nn": ${N_BLOCKS_NN}
}
EOF

  # Define script variables
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR=${SCRIPT_DIR}/
  APP_CLASSPATH=${SCRIPT_DIR}/
  APP_PYTHONPATH=${SCRIPT_DIR}/

  # Define application variables
  graph=$tracing
  log_level="off"
  qos_flag=${QOS_FLAG:---qos=debug}
  # Aggressively limit the memory of the workers
  workers_flag="--jvm_workers_opts=\"-Xms1000m,-Xmx4000m,-Xmn500m"\"
  constraints=""

  WORKER_IN_MASTER=0

  export DISLIB_TO_USE=~/dislib
  export USE_DATACLAY=1

  if [ "$COMPSS_WORKING_DIR" = "local_disk" ]; then
    working_dir_params="--worker_working_dir=local_disk"
  else
    # $USER is "built-in", $GROUP is not
    GROUP=$(id -g -n $USER)
    COMPSS_WORKING_DIR=/gpfs/scratch/$GROUP/$USER
    working_dir_params="--worker_working_dir=$COMPSS_WORKING_DIR --master_working_dir=$COMPSS_WORKING_DIR --base_log_dir=$COMPSS_WORKING_DIR"
  fi

  export ComputingUnits=${COMPUTING_UNITS}

  # Enqueue job
  enqueue_compss \
    --job_name=nn-split \
    --exec_time="${execution_time}" \
    --num_nodes="${num_nodes}" \
    \
    --cpus_per_node="${CPUS_PER_NODE}" \
    --worker_in_master_cpus="${WORKER_IN_MASTER}" \
    --scheduler=$COMPSS_SCHEDULER \
    \
    "${workers_flag}" \
    \
    $working_dir_params \
    \
    --constraints=${constraints} \
    --tracing="${tracing}" \
    --graph="${graph}" \
    --summary \
    --log_level="${log_level}" \
    "${qos_flag}" \
    \
    --classpath=${DATACLAY_JAR} \
    --pythonpath=${PYCLAY_PATH}:${APP_PYTHONPATH}:${DISLIB_TO_USE}:/gpfs/home/bsc25/bsc25865/tad4bj/src:${PYTHONPATH} \
    --storage_props=${STORAGE_PROPS} \
    --storage_home=$COMPSS_STORAGE_HOME \
    --prolog="tad4bj,setnow,start_ts" \
    --prolog="tad4bj,setdict,$TAD4BJ_JSON" \
    --prolog="$DATACLAY_HOME/bin/dataclayprepare,$(pwd)/../dislib_model/,$(pwd),dislib_model,python" \
    --epilog="tad4bj,setnow,finish_ts" \
    \
    ${extra_tracing_flags} \
    \
    --lang=python \
    \
    "$(pwd)/nn.py"


# To avoid the clean up of the working directory add this flag:
#--jvm_workers_opts="-Dcompss.worker.removeWD=false" \
