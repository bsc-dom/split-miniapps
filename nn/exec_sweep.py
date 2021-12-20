#!/usr/bin/env python

import os
import subprocess

EXECUTION_VALUES_FILE = "execution_values"
STORAGE_PROPS_FILE = "cfgfiles/storage_props.cfg"


def build_exec_values(points_per_block, n_blocks_fit, n_blocks_nn,
                      number_of_steps=5, extra_args=None):
    with open(EXECUTION_VALUES_FILE, "w") as f:
        f.write("""
export POINTS_PER_BLOCK=%d
export N_BLOCKS_FIT=%d
export N_BLOCKS_NN=%d
export NUMBER_OF_STEPS=%d
""" % (points_per_block, n_blocks_fit, n_blocks_nn, number_of_steps))

        if extra_args:
            # At this point extra_args is neither None nor empty, assuming it is a populated dict
            for variable, value in extra_args.items():
                f.write("export %s=%s\n" % (variable.upper(), value))


def build_storage_props(backends_per_node=2, cpus_per_node=48, computing_units=4):
    with open(STORAGE_PROPS_FILE, "w") as f:
        f.write("""
BACKENDS_PER_NODE=%d
CPUS_PER_NODE=%d
COMPUTING_UNITS=%d
""" % (backends_per_node, cpus_per_node, computing_units))


def round_of_execs(points_per_block, n_blocks_fit, n_blocks_nn,
                   number_of_nodes=3, execution_time=60, tracing=False, clear_qos_flag=True,
                   extra_args=None):

    print("Sending execution to %d workers.\n" 
            "Total blocks:\n"
            "\t#%d blocks per fit\n"
            "\t#%d blocks per NN\n"
            "Points per block: #%d"
        % (n_workers, n_blocks_fit, n_blocks_nn, points_per_block))

    newenv = dict(os.environ)
    if clear_qos_flag:
        newenv["QOS_FLAG"] = " "

    build_exec_values(points_per_block, n_blocks_fit, n_blocks_nn,
                      extra_args=extra_args)
    subprocess.call("./launch_without_dataClay.sh %d %d %s" 
                    % (number_of_nodes, execution_time, str(tracing).lower()),
                    shell=True, env=newenv)

    build_exec_values(points_per_block, n_blocks_fit, n_blocks_nn,
                      extra_args=extra_args)
    subprocess.call("./launch_with_dataClay.sh %d %d %s" 
                    % (number_of_nodes, execution_time, str(tracing).lower()),
                    shell=True, env=newenv)


if __name__ == "__main__":

    # Common storage properties
    build_storage_props()

    points_per_block = 1000000

    for i in range(5):
        n_workers = 2 ** i
        n_blocks_fit = 96 * n_workers
        n_blocks_nn = 8 * n_workers

        round_of_execs(points_per_block, n_blocks_fit, n_blocks_nn,
                       number_of_nodes=n_workers + 1, execution_time=90)

    for i in range(5):
        n_workers = 2 ** i
        n_blocks_fit = 48 * n_workers
        n_blocks_nn = 48 * n_workers

        round_of_execs(points_per_block, n_blocks_fit, n_blocks_nn,
                       number_of_nodes=n_workers + 1, execution_time=240)
