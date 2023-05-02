#!/usr/bin/env python

import re
import os
import subprocess
import itertools

EXECUTION_VALUES_FILE = "execution_values"
STORAGE_PROPS_FILE = "cfgfiles/storage_props.cfg"

# The flag is the literal string None for enqueue_compss
LAST_GPFS_JOB = "25442352"

# Schedulers:
FIFODLOCS = "es.bsc.compss.scheduler.fifodatalocation.FIFODataLocationScheduler"
FIFODS = "es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler"

COMPSS_ALTERNATIVE = {
    "compss_scheduler": FIFODS,
    "compss_working_dir": "gpfs"
}


def build_exec_values(points_per_block, n_blocks_fit, n_blocks_nn, copy_fit_struct=1,
                      compss_scheduler=FIFODLOCS,
                      compss_working_dir="local_disk",
                      use_split=None, dask_rechunk=0,
                      number_of_steps=5, extra_args=None):
    with open(EXECUTION_VALUES_FILE, "w") as f:
        f.write("""
export POINTS_PER_BLOCK=%d
export N_BLOCKS_FIT=%d
export N_BLOCKS_NN=%d
export NUMBER_OF_STEPS=%d
export COPY_FIT_STRUCT=%d

export COMPSS_SCHEDULER=%s
export COMPSS_WORKING_DIR=%s
export DASK_RECHUNK=%d
""" % (points_per_block, n_blocks_fit, n_blocks_nn, number_of_steps, copy_fit_struct,
       compss_scheduler, compss_working_dir, dask_rechunk))

        if use_split is not None:
            f.write("export USE_SPLIT=%d\n" % int(use_split))

        if extra_args:
            # At this point extra_args is neither None nor empty, assuming it is a populated dict
            for variable, value in extra_args.items():
                f.write("export %s=%s\n" % (variable.upper(), value))


def build_storage_props(backends_per_node=2, cpus_per_node=48, computing_units=1):
    with open(STORAGE_PROPS_FILE, "w") as f:
        f.write("""
BACKENDS_PER_NODE=%d
CPUS_PER_NODE=%d
COMPUTING_UNITS=%d
""" % (backends_per_node, cpus_per_node, computing_units))


sbj_jobid = re.compile(r"Submitted batch job (\d+)", re.MULTILINE)

def process_completed_job(completed_process):
    """Store the output into a file for debugging, while also return the jobid."""    
    m = re.search(sbj_jobid, completed_process.stdout.decode("ascii"))

    if m is None:
        with open("submission.out", "wb") as f:
            f.write(completed_process.stdout)
        with open("submission.err", "wb") as f:
            f.write(completed_process.stderr)
        raise SystemError("Submission did not return a jobid. Dumping stdout/stderr on submission.{out,err}")

    jobid = m[1]

    print("Submission of job %s has been finished" % jobid)

    with open("%s_submit.out" % jobid, "wb") as f:
        f.write(completed_process.stdout)

    with open("%s_submit.err" % jobid, "wb") as f:
        f.write(completed_process.stderr)

    return jobid


def round_of_execs(points_per_block, n_blocks_fit, n_blocks_nn,
                   number_of_nodes=3, execution_time=60, tracing=False, clear_qos_flag=True,
                   extra_args=None):

    global LAST_GPFS_JOB

    print("Executing in #%d workers (nodes=%d)" % (number_of_nodes-1, number_of_nodes))
    print("Total blocks:\n"
            "\t#%d blocks per fit\n"
            "\t#%d blocks per NN\n"
            "Points per block: #%d"
        % (n_blocks_fit, n_blocks_nn, points_per_block))

    newenv = dict(os.environ)
    if clear_qos_flag:
        newenv["QOS_FLAG"] = " "

    if extra_args is None:
        extra_args = dict()

    dask_rechunk = n_blocks_fit // (2 * (number_of_nodes - 1))

    for dask_options in [{"use_split": False}, {"use_split": True}, {"use_split": False, "dask_rechunk": dask_rechunk}] * 3:
        build_exec_values(points_per_block, n_blocks_fit, n_blocks_nn, copy_fit_struct=1,
                          extra_args=extra_args, **dask_options)
        cp = subprocess.run("./launch_with_dask.sh %d %d %s"
                            % (number_of_nodes, execution_time, str(tracing).lower()),
                            shell=True, env=newenv, capture_output=True)
        process_completed_job(cp)

        build_exec_values(points_per_block, n_blocks_fit, n_blocks_nn, copy_fit_struct=0,
                          extra_args=extra_args, **dask_options)
        cp = subprocess.run("./launch_with_dask.sh %d %d %s"
                            % (number_of_nodes, execution_time, str(tracing).lower()),
                            shell=True, env=newenv, capture_output=True)
        process_completed_job(cp)

    # for use_split in [0, 1]:
    #     ea_to_use = extra_args.copy()
    #     ea_to_use["use_split"] = use_split

    #     build_exec_values(points_per_block, n_blocks_fit, n_blocks_nn, copy_fit_struct=1,
    #                       extra_args=ea_to_use)
    #     cp = subprocess.run("./launch_with_dataClay.sh %d %d %s" 
    #                         % (number_of_nodes, execution_time, str(tracing).lower()),
    #                         shell=True, env=newenv, capture_output=True)
    #     process_completed_job(cp)

    # newenv["JOB_DEPENDENCY"] = LAST_GPFS_JOB

    # build_exec_values(points_per_block, n_blocks_fit, n_blocks_nn,
    #                   extra_args=extra_args, 
    #                   **COMPSS_ALTERNATIVE)
    # cp = subprocess.run("./launch_without_dataClay.sh %d %d %s" 
    #                     % (number_of_nodes, execution_time, str(tracing).lower()),
    #                     shell=True, env=newenv, capture_output=True)

    # LAST_GPFS_JOB = process_completed_job(cp)
    # print("Using %s for GPFS dependency (COMPSs executions)" % LAST_GPFS_JOB)


if __name__ == "__main__":

    # Common storage properties
    build_storage_props()

    points_per_block = 500000

    for i, fit_per_worker in itertools.product([0, 1, 2, 3, 4], [2, 6]):
        n_workers = 2 ** i
        n_blocks_fit = fit_per_worker * n_workers
        n_blocks_nn = 24 * n_workers

        round_of_execs(points_per_block, n_blocks_fit, n_blocks_nn,
                       number_of_nodes=n_workers + 1, execution_time=10+10 * n_workers)

    for fit_per_worker in [2, 4, 6, 8, 10, 12]:
        n_workers = 8
        n_blocks_fit = fit_per_worker * n_workers
        n_blocks_nn = 24 * n_workers

        round_of_execs(points_per_block, n_blocks_fit, n_blocks_nn,
                       number_of_nodes=n_workers + 1, execution_time=30 + 10 * fit_per_worker)
                       #execution_time=10)


    ######################################################################################
    # Everything that follows I believe that does not make sense for comparison
    # (because it affects more the fit than the training and there is not enough data)
    ######################################################################################

    # # Worst case-ish
    # for i in range(5):
    #     n_workers = 2 ** i
    #     n_blocks_fit = 2 * n_workers
    #     n_blocks_nn = 48
    #     round_of_execs(points_per_block, n_blocks_fit, n_blocks_nn,
    #                    number_of_nodes=n_workers + 1, execution_time=10 + 10 * n_workers)

    #     round_of_execs(points_per_block, n_blocks_fit, n_blocks_nn,
    #                    number_of_nodes=n_workers + 1, execution_time=10 + 10 * n_workers)

    #     round_of_execs(points_per_block, n_blocks_fit, n_blocks_nn,
    #                    number_of_nodes=n_workers + 1, execution_time=10 + 10 * n_workers)
