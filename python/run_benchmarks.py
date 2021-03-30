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

# Get amount of free system memory, return amount in Gb
def getFreeMem():
    free_mem_in_kb = ""
    inactive_mem_in_kb = ""
    free_mem_in_Gb = 0
    with open('/proc/meminfo') as file:
        for line in file:
            if 'MemFree' in line:
                free_mem_in_kb = line.split()[1]
            if 'Inactive' in line:
                inactive_mem_in_kb = line.split()[1]
    if(free_mem_in_kb is "") or (inactive_mem_in_kb is ""):
        raise Exception("Could not retreive amount of free system memory")
    free_mem_in_Gb = (int(free_mem_in_kb) + int(inactive_mem_in_kb)) / (2**20)
    return free_mem_in_Gb

# return required host memory for FDD in Gb
def hostMemoryRequiredFDD(nsamps, nchans, dm_count):
    # sizeof(dedisp_float)=4; sizeof(dedisp_byte) = 1
    # input buffer: nsamps * nchans * sizeof(dedisp_float)
    h_in_buffer = nsamps * nchans * 4
    # output buffer: nsamps * dm_count * (out_nbits/8) * sizeof(dedisp_byte)
    h_out_buffer = nsamps * dm_count * 4
    h_implementation = h_in_buffer + h_out_buffer
    # total required host memory = application host memory + implementation host memory
    # bench.hpp uses a shared input/output dummy buffer
    # we take the max for either the input or output buffer
    h_application = max(h_in_buffer, h_out_buffer)
    h_total = h_application + h_implementation
    h_total_margin = h_total * 1.05 #include 5% margin
    h_total_margin_Gb = h_total_margin / (2**30)
    return h_total_margin_Gb

# Run application and capture output
if __name__ == "__main__":
    # Get parsed arguments
    argParser = create_arg_parser()
    parsedArgs = argParser.parse_args(sys.argv[1:])

    # How many times to repeat each test
    niterations = int(parsedArgs.niterations)

    executableDirectory = parsedArgs.executableDirectory
    if executableDirectory and not os.path.exists(parsedArgs.executableDirectory):
        raise Exception("Directory" + parsedArgs.executableDirectory + "does not exist")

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

    # Get current free system memory in Gb
    system_mem_free = getFreeMem()
    print(f'Current system memory free: {system_mem_free:.2f}G')
    # Check it there is at least 80 Gb of free memory,
    # this is a sensible amount for the parameters that we use
    if(system_mem_free < 80.0): raise Exception(f'There is not sufficient system memory available to run the benchmarks (free memory: {system_mem_free:.2f}G)')
    # For FDD we add an aditional parameter specific check (below).
    # FDD is most critical out of the three different implementations.
    # Checking might be added for TDD and dedisp as well.
    # However, those implementations have many more parameters to take in to account,
    # maybe an approximation might suffice.

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

                            # Check if there is sufficient memory available for FDD
                            if (benchmark is "fdd"):
                                if (device is "GPU"):
                                    system_mem_required = hostMemoryRequiredFDD(nsamp, nchan, ndm)
                                    if(system_mem_required > system_mem_free):
                                        print(f'Skipping {device}_{benchmark}_nchan{nchan}_nsamp{nsamp}_ndm{ndm} because there is not sufficient system memory available (free memory: {system_mem_free:.2f}G while required: {system_mem_required:.2f}G)')
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