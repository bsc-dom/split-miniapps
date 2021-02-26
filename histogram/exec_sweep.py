#!/usr/bin/env python

import os
import subprocess

EXECUTION_VALUES_FILE = "execution_values"
STORAGE_PROPS_FILE = "cfgfiles/storage_props.cfg"


def build_exec_values(points_per_fragment, number_of_fragments, 
                      use_split=None, roundrobin_persistence=None,
                      use_reduction_decorator=None, extra_args=None):
    with open(EXECUTION_VALUES_FILE, "w") as f:
        f.write("""
export POINTS_PER_FRAGMENT=%d
export NUMBER_OF_FRAGMENTS=%d
""" % (points_per_fragment, number_of_fragments))
        if use_split is not None:
            f.write("export USE_SPLIT=%d\n" % int(use_split))
        
        if roundrobin_persistence is not None:
            f.write("export ROUNDROBIN_PERSISTENCE=%d\n" % int(roundrobin_persistence))

        if use_reduction_decorator is not None:
            f.write("export USE_REDUCTION_DECORATOR=%d\n" % int(use_reduction_decorator))

        if extra_args:
            # At this point extra_args is neither None nor empty, assuming it is a populated dict
            for variable, value in extra_args.items():
                f.write("export %s=%s\n" % (variable.upper(), value))


def build_storage_props(backends_per_node=48, cpus_per_node=48):
    with open(STORAGE_PROPS_FILE, "w") as f:
        f.write("""
BACKENDS_PER_NODE=%d
CPUS_PER_NODE=%d
""" % (backends_per_node, cpus_per_node))


def round_of_execs(points_per_fragment, number_of_fragments,
                   number_of_nodes=3, execution_time=40, tracing=False, clear_qos_flag=True,
                   extra_args=None):

    newenv = dict(os.environ)
    if clear_qos_flag:
        newenv["QOS_FLAG"] = " "

    build_exec_values(points_per_fragment, number_of_fragments,
                      use_reduction_decorator=True, 
                      extra_args=extra_args)
    subprocess.call("./launch_without_dataClay.sh %d %d %s" 
                    % (number_of_nodes, execution_time, str(tracing).lower()),
                    shell=True, env=newenv)

    build_exec_values(points_per_fragment, number_of_fragments,
                      use_reduction_decorator=False, 
                      extra_args=extra_args)
    subprocess.call("./launch_without_dataClay.sh %d %d %s" 
                    % (number_of_nodes, execution_time, str(tracing).lower()),
                    shell=True, env=newenv)

    build_exec_values(points_per_fragment, number_of_fragments,
                      use_split=False, roundrobin_persistence=True,
                      use_reduction_decorator=True, 
                      extra_args=extra_args)
    subprocess.call("./launch_with_dataClay.sh %d %d %s" 
                    % (number_of_nodes, execution_time, str(tracing).lower()),
                    shell=True, env=newenv)

    build_exec_values(points_per_fragment, number_of_fragments,
                      use_split=False, roundrobin_persistence=True,
                      use_reduction_decorator=False,
                      extra_args=extra_args)
    subprocess.call("./launch_with_dataClay.sh %d %d %s" 
                    % (number_of_nodes, execution_time, str(tracing).lower()),
                    shell=True, env=newenv)

    build_exec_values(points_per_fragment, number_of_fragments,
                      use_split=True, roundrobin_persistence=True,
                      use_reduction_decorator=True,
                      extra_args=extra_args)
    subprocess.call("./launch_with_dataClay.sh %d %d %s" 
                    % (number_of_nodes, execution_time, str(tracing).lower()),
                    shell=True, env=newenv)

    build_exec_values(points_per_fragment, number_of_fragments,
                      use_split=True, roundrobin_persistence=True,
                      use_reduction_decorator=False,
                      extra_args=extra_args)
    subprocess.call("./launch_with_dataClay.sh %d %d %s" 
                    % (number_of_nodes, execution_time, str(tracing).lower()),
                    shell=True, env=newenv)


if __name__ == "__main__":
    # Common storage properties
    build_storage_props()

    for i in range(8):
        number_of_fragments = (2 ** i) * 96
        points_per_fragment = 2 ** (10 - i) * 100000
        print("Sending execution with #%d fragments (n_points=%d)" % 
              (number_of_fragments, points_per_fragment))

        round_of_execs(points_per_fragment, number_of_fragments)
