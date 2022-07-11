#!/usr/bin/env python

import re
import os
import subprocess

EXECUTION_VALUES_FILE = "execution_values"
STORAGE_PROPS_FILE = "cfgfiles/storage_props.cfg"

BASE_POINTS_PER_FRAGMENT = 128
BASE_NUMBER_OF_FRAGMENTS = 384  # 48 * 8
BIG_POINTS_PER_FRAGMENT = BASE_POINTS_PER_FRAGMENT * 8

# The flag is the literal string None for enqueue_compss
LAST_GPFS_JOB = "None"

# Schedulers:
FIFODLOCS = "es.bsc.compss.scheduler.fifodatalocation.FIFODataLocationScheduler"
FIFODS = "es.bsc.compss.scheduler.fifodatanew.FIFODataScheduler"

COMPSS_ALTERNATIVE = {
    "compss_scheduler": FIFODS,
    "compss_working_dir": "gpfs"
}

def build_exec_values(points_per_fragment, number_of_fragments,
                      compss_scheduler=FIFODLOCS,
                      compss_working_dir="local_disk",
                      use_split=None, extra_args=None):
    with open(EXECUTION_VALUES_FILE, "w") as f:
        f.write("""
export POINTS_PER_FRAGMENT=%d
export NUMBER_OF_FRAGMENTS=%d

export COMPSS_SCHEDULER=%s
export COMPSS_WORKING_DIR=%s
""" % (points_per_fragment, number_of_fragments,
       compss_scheduler, compss_working_dir))

        if use_split is not None:
            f.write("export USE_SPLIT=%d\n" % int(use_split))

        if extra_args:
            # At this point extra_args is neither None nor empty, assuming it is a populated dict
            for variable, value in extra_args.items():
                f.write("export %s=%s\n" % (variable.upper(), value))


def build_storage_props(backends_per_node=2, cpus_per_node=48):
    with open(STORAGE_PROPS_FILE, "w") as f:
        f.write("""
BACKENDS_PER_NODE=%d
CPUS_PER_NODE=%d
""" % (backends_per_node, cpus_per_node))


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

def round_of_execs(points_per_fragment, number_of_fragments,
                   number_of_nodes=3, execution_time=40, tracing=False, clear_qos_flag=True,
                   extra_args=None):

    global LAST_GPFS_JOB

    print("Executing in #%d workers (nodes=%d)" % (number_of_nodes-1, number_of_nodes))
    print("Executing with #%d fragments, (n_points=%d)"
        % (number_of_fragments, points_per_fragment))

    newenv = dict(os.environ)
    if clear_qos_flag:
        newenv["QOS_FLAG"] = " "

    build_exec_values(points_per_fragment, number_of_fragments, use_split=False,
                      extra_args=extra_args)

    cp = subprocess.run("./launch_with_dataClay.sh %d %d %s"
                        % (number_of_nodes, execution_time, str(tracing).lower()),
                        shell=True, env=newenv, capture_output=True)

    process_completed_job(cp)

    build_exec_values(points_per_fragment, number_of_fragments, use_split=True,
                      extra_args=extra_args)

    cp = subprocess.run("./launch_with_dataClay.sh %d %d %s"
                        % (number_of_nodes, execution_time, str(tracing).lower()),
                        shell=True, env=newenv, capture_output=True)

    process_completed_job(cp)

    newenv["JOB_DEPENDENCY"] = LAST_GPFS_JOB
    build_exec_values(points_per_fragment, number_of_fragments,
                      use_split=False,
                      extra_args=extra_args, 
                      **COMPSS_ALTERNATIVE)
    cp = subprocess.run("./launch_without_dataClay.sh %d %d %s" 
                        % (number_of_nodes, execution_time, str(tracing).lower()),
                        shell=True, env=newenv, capture_output=True)

    LAST_GPFS_JOB = process_completed_job(cp)
    print("Using %s for GPFS dependency (COMPSs executions)" % LAST_GPFS_JOB)


if __name__ == "__main__":

    # Common storage properties
    build_storage_props()

    print()
    print("*** Weak scaling")
    for i in range(5):
        workers = 2 ** i
        number_of_fragments = BASE_NUMBER_OF_FRAGMENTS * workers
        points_per_fragment = BASE_POINTS_PER_FRAGMENT
        round_of_execs(points_per_fragment, number_of_fragments,
                       number_of_nodes=workers+1, execution_time=20+10*workers)

    print()
    print("*** Weak scaling with big blocks")
    for i in range(5):
        workers = 2 ** i
        number_of_fragments = 48 * workers
        points_per_fragment = BIG_POINTS_PER_FRAGMENT
        round_of_execs(points_per_fragment, number_of_fragments,
                       number_of_nodes=workers+1, execution_time=20+10*workers)

    print()
    print("*** Blocksize sweep for 8 nodes")
    for granularity in [1, 2, 4, 8]:
        workers = 8
        number_of_fragments = 48 * workers * granularity
        points_per_fragment = BIG_POINTS_PER_FRAGMENT // granularity
        round_of_execs(points_per_fragment, number_of_fragments,
                       number_of_nodes=workers+1, execution_time=120)
