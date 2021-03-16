#!/usr/bin/env python3
# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later
# Script to run muliple consecutive benchmarks and store output.
# run_benchmarks_analysis.py might be used to analyze the ouput files.
# Note: requires "DEDISP_BENCHMARK" to be defined with cmake

import subprocess
import re
import numpy as np
import time
import tqdm #progress bar
import sys
import os
import argparse

# Creates and returns the ArgumentParser object
def create_arg_parser():
    parser = argparse.ArgumentParser(description='Batch process benchmarks and store output')
    parser.add_argument('executableDirectory', nargs='?', default="",
                    help='The directory with the executables for the benchmarks, e.g. ...bin/dedisp/bin/')
    parser.add_argument('--niterations', nargs='?', default=5,
                    help='How many times to repeat each test')
    #parser.add_argument('--GPU',
    #                help='Which GPU to use')
    # would be nice to implement, should also include binding to the socket that connects to the GPU
    parser.add_argument('--dryRun', action='store_true',
                    help='Dry Run')
    # Use this if you want to test the parameter space and directory and file creation
    return parser

# Print a dict with formatting
def prettyPrintDict(dictToPrint):
    for key in dictToPrint:
        printString = '{:45} {} '.format(key, dictToPrint[key])
        print(printString)
    return

# Run application and capture output
if __name__ == "__main__":
    # Get parsed arguments
    argParser = create_arg_parser()
    parsedArgs = argParser.parse_args(sys.argv[1:])

    # How many times to repeat each test
    niterations = int(parsedArgs.niterations)

    executableDirectory = parsedArgs.executableDirectory
    if executableDirectory and not os.path.exists(parsedArgs.executableDirectory):
        raise("Directory" + parsedArgs.executableDirectory + "does not exist")

    # if (parsedArgs.GPU):
    # fixme, should also include binding to the socket that connects to the GPU
    #     print(f'Is CUDA_VISIBILE_DEVICES already set? {int(os.environ.get('CUDA_VISIBILE_DEVICES', 'Not Set'))}'')
    #     print(f'Set CUDA_VISIBILE_DEVICES to {str(parsedArgs.GPU)}')
    #     os.environ['CUDA_VISIBILE_DEVICES'] = str(parsedArgs.GPU)
    #     print(f'Is CUDA_VISIBILE_DEVICES now set? {int(os.environ.get('CUDA_VISIBILE_DEVICES', 'Not Set'))}'')

    # Which tests to run
    # mytests is a dictionay of benchmark names and executable commands
    # Use full paths in the commands
    # No spaces in mytest name!
    # For reference
    string_numactl   =   "numactl --cpunodebind=0 " #fixed for now, might be made configurable
    path_bench       =   os.path.join(executableDirectory, "bench")

    mytests = dict()

    # Sensible parameters for benchmarking
    # We need some 'data volume' to actually keep te GPU busy
    # and see the effects of batch processing of data
    # Note: usage of RAM quickly increases with larger numbers!
    parameters_benchmark = { "dedisp", "tdd", "fdd" }
    parameters_device = { "GPU" } #{ "CPU", "GPU" }
    parameters_nchan = {1024} # {1024, 2048, 4096}
    parameters_nsamp =  {4689920} #5 minutes # { 4689920, 9379840, 14069760} # 5, 10, 15 minutes
    parameters_segmented = { False } # { True, False }
    parameters_ndm = { 128, 256, 512, 1024, 2048, 4096 }

    # Create a dict with testnames and commands to execute the test
    for benchmark in parameters_benchmark:
        for device in parameters_device:
            for nchan in parameters_nchan:
                for segmented in parameters_segmented:
                    for nsamp in parameters_nsamp:
                        for ndm in parameters_ndm:

                            # Skip certain combination of parameters
                            if (benchmark is not "fdd"):
                                # Only run CPU benchmark for FDD
                                if (device is "CPU"):
                                    continue
                                # Only run segmented benchmark for FDD
                                if (segmented):
                                    continue

                            if (benchmark is "fdd"):
                                if (device is "GPU"):
                                    # Skip ndm > 3000 (bug to be fixed)
                                    if (int(ndm) > 3000):
                                        continue

                            # Set environment variables
                            if (device is "GPU"):
                                # Use numactl for all GPU benchmarks
                                executable = string_numactl + path_bench + benchmark
                                environment = "USE_CPU=0"
                            else:
                                executable = path_bench + benchmark
                                environment = "USE_CPU=1"

                            # Handle fdd segmented
                            suffix = ""
                            if (benchmark is "fdd"):
                                environment += f" USE_SEGMENTED={int(segmented)}"
                                if (segmented):
                                    suffix += "-seg"

                            name = f"{device}_{benchmark}{suffix}_nchan{nchan}_nsamp{nsamp}_ndm{ndm}"
                            command = f"{environment} {executable} -s {int(nsamp)} -n {ndm} -c {nchan} -i {niterations}"

                            # Add test
                            test = { name, command }
                            mytests[name] = command

    # Get start times and format them
    starttime=time.localtime()
    totaltime = time.time()
    print(f'Starting application at {time.strftime("%a %b %d %H:%M:%S %Y",starttime)}')

    # Check paths, create directory, set path variables
    # Create this structure:
    # <current path>/bench_results_<current datetime>/
    # For each testentry in mytests
    #   <testentry>.txt
    directory_datetime = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
    directory = 'bench_results_'+directory_datetime
    try:
        os.mkdir(directory)
    except OSError as error:
        print(error)
    print(f'Created directory {directory} for bechmark run results')

    print(f'Found {len(mytests)} tests to run:')
    prettyPrintDict(mytests)

    # Loop over all tests
    for i, testentry in enumerate(mytests):
        print(f'### {time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())} Running test: {i+1} out of {len(mytests)}')
        print(f'{testentry}')

        # Create file to store output
        path = os.path.join(directory,testentry+".txt")
        try:
            f = open(path, "w")
        except OSError:
            print(f'Could not open file: {path}')
            break
        f.write(f'### Benchmark: {testentry}, command {mytests[testentry]} ###\n')

        # Run the application and capture output or skip running the actual application for a dry run
        if(not parsedArgs.dryRun):
            output = subprocess.getoutput(mytests[testentry])
        else: output = '...'
        # Save to file
        f.write(output)

        f.close()

    # Wrap up
    totaltime = time.time()-totaltime
    print('### Ending application at {} processing took {:.2f} seconds'.format(time.strftime("%a %b %d %H:%M:%S %Y",time.localtime()),totaltime))