#!/bin/bash -e

  # Retrieve script arguments
  num_nodes=${1:-9}
  execution_time=${2:-60}
  tracing=${3:-true}

  if [[ $tracing == "true" ]]  ; then
    TRACING_FLAG=1
  else
    TRACING_FLAG=0
  fi

  exec_values_cp=`mktemp -p ~`

  qos_flag=${QOS_FLAG:---qos=debug}

  source execution_values
  cp execution_values $exec_values_cp
  sbatch_file=`mktemp -p ~`

  TAD4BJ_JSON=`mktemp -p ~`
  cat << EOF > $TAD4BJ_JSON
{
  "dask": 1,
  "tracing": ${TRACING_FLAG},
  "nodes": ${num_nodes},
  "num_threads": ${OPENBLAS_NUM_THREADS:-48},
  "use_split": ${USE_SPLIT},
  "copy_fit_struct": ${COPY_FIT_STRUCT},
  "points_per_block": ${POINTS_PER_BLOCK},
  "n_blocks_fit": ${N_BLOCKS_FIT},
  "n_blocks_nn": ${N_BLOCKS_NN},
  "dask_rechunk": ${DASK_RECHUNK}
}
EOF

  cat << EOF > $sbatch_file
#!/bin/bash
#SBATCH --nodes=$num_nodes
#SBATCH --time=$execution_time
#SBATCH --job-name=nn-split
#SBATCH $qos_flag

export PYTHONPATH=$PYTHONPATH:/gpfs/home/bsc25/bsc25865/tad4bj/src

echo ""
echo "********************"
echo "Calling dask-prepare"
echo "********************"
$PWD/../dask-prepare.sh

echo ""
echo "********************"
echo "Dask should be ready"
echo "********************"

sleep 10

set -a
source $exec_values_cp

echo ""
echo "**************"
echo "Ready to start"
echo "**************"

tad4bj setnow start_ts
tad4bj setdict $TAD4BJ_JSON

echo " * * * nn-dask.py * * * "
~/anaconda3/bin/python -u $PWD/nn-dask.py

echo ""
echo "********"
echo "Finished"
echo "********"

tad4bj setnow finish_ts

EOF

  echo "Ready to run sbatch on file $sbatch_file"
  sbatch $sbatch_file
