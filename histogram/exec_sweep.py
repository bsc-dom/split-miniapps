#!/usr/bin/env python

import subprocess

EXECUTION_VALUES_FILE = "execution_values"
STORAGE_PROPS_FILE = "cfgfiles/storage_props.cfg"


def build_exec_values(points_per_fragment, number_of_fragments, use_split=None, roundrobin_persistence=None):
    with open(EXECUTION_VALUES_FILE, "w") as f:
        f.write("""
export POINTS_PER_FRAGMENT=%d
export NUMBER_OF_FRAGMENTS=%d
""" % (points_per_fragment, number_of_fragments))
        if use_split is not None:
            f.write("export USE_SPLIT=%d\n" % int(use_split))
        
        if roundrobin_persistence is not None:
            f.write("export ROUNDROBIN_PERSISTENCE=%d\n" % int(roundrobin_persistence))


def build_storage_props(backends_per_node=48, cpus_per_node=48):
    with open(STORAGE_PROPS_FILE, "w") as f:
        f.write("""
BACKENDS_PER_NODE=%d
CPUS_PER_NODE=%d
""" % (backends_per_node, cpus_per_node))


if __name__ == "__main__":
    for i in range(8):
        build_storage_props(24, 24)
        number_of_fragments = (2 ** i) * 48
        points_per_fragment = 2 ** (27 - i) * 2
        print("Sending execution with #%d fragments (n_points=%d)" % 
              (number_of_fragments, points_per_fragment))

        build_exec_values(points_per_fragment, number_of_fragments)
        subprocess.call("./launch_without_dataClay.sh", shell=True)

        build_exec_values(points_per_fragment, number_of_fragments, 
                          use_split=True, roundrobin_persistence=True)
        subprocess.call("./launch_with_dataClay.sh", shell=True)

        build_exec_values(points_per_fragment, number_of_fragments,
                          use_split=False, roundrobin_persistence=True)
        subprocess.call("./launch_with_dataClay.sh", shell=True)

        build_storage_props(48, 48)
        number_of_fragments = (2 ** i) * 96
        points_per_fragment = 2 ** (27 - i) * 1
        print("Sending execution with #%d fragments (n_points=%d)" % 
              (number_of_fragments, points_per_fragment))

        build_exec_values(points_per_fragment, number_of_fragments)
        subprocess.call("./launch_without_dataClay.sh", shell=True)

        build_exec_values(points_per_fragment, number_of_fragments, 
                          use_split=True, roundrobin_persistence=True)
        subprocess.call("./launch_with_dataClay.sh", shell=True)

        build_exec_values(points_per_fragment, number_of_fragments,
                          use_split=False, roundrobin_persistence=True)
        subprocess.call("./launch_with_dataClay.sh", shell=True)
