#!/usr/bin/env python3
#Script to run muliple consecutive benchmarks and store output

import subprocess
import re
import numpy as np
import time
import tqdm #progress bar
import sys
import os

import argparse

def create_arg_parser():
    # Creates and returns the ArgumentParser object
    parser = argparse.ArgumentParser(description='Batch process benchmarks and store output')
    parser.add_argument('executableDirectory',
                    help='The directory with the executables for the benchmarks, e.g. ...bin/dedisp/bin/')
    parser.add_argument('--niterations',
                    help='Optional the number of iterations')
    parser.add_argument('--GPU',
                    help='Which GPU to use')
    return parser

if __name__ == "__main__":
    #Run application and capture output

    argParser = create_arg_parser()
    parsedArgs = argParser.parse_args(sys.argv[1:])

    #How many times to repeat each test
    if(parsedArgs.niterations): niterations = parsedArgs.niterations
    else: niterations = 5 #default

    if os.path.exists(parsedArgs.executableDirectory):
        executableDirectory = parsedArgs.executableDirectory
    else: #raise exception
        raise("Directory" + parsedArgs.executableDirectory + "does not exist")

    # if (parsedArgs.GPU):
    # fixme
    #     print(f'Is CUDA_VISIBILE_DEVICES already set? {int(os.environ.get('CUDA_VISIBILE_DEVICES', 'Not Set'))}'')
    #     print(f'Set CUDA_VISIBILE_DEVICES to {str(parsedArgs.GPU)}')
    #     os.environ['CUDA_VISIBILE_DEVICES'] = str(parsedArgs.GPU)
    #     print(f'Is CUDA_VISIBILE_DEVICES now set? {int(os.environ.get('CUDA_VISIBILE_DEVICES', 'Not Set'))}'')

    #Which tests to run
    # mytests is a dictionay of benchmark names and executable commands
    # Use full paths in the commands
    # No spaces in mytest name!
    # For reference
    string_numactl          =   "numactl --cpunodebind=0 "
    path_benchfdd           =   os.path.join(executableDirectory, "benchfdd ")
    path_benchdedisp        =   os.path.join(executableDirectory, "benchdedisp ")
    string_use_cpu          =   " USE_CPU=1 "
    string_use_cpu_ref      =   " USE_CPU=1 USE_REFERENCE=1 "
    string_use_segmented    =   " USE_SEGMENTED=1 "
    string_nsamp24          =   " -s 240000 "
    string_nsamp48          =   " -s 480000 "
    string_default_param    =   string_nsamp24
    mytests_stub = {    "Test1": os.path.join(executableDirectory, "print.sh 1"),
                        "Test2": os.path.join(executableDirectory, "print.sh 2"),
                        "Test3": os.path.join(executableDirectory, "print.sh 3")}
    mytest_CPU_ref =    {   "CPU_reference"                     : string_use_cpu_ref + path_benchfdd + string_default_param,
                            "CPU_reference_segmented"           : string_use_cpu_ref + string_use_segmented + path_benchfdd + string_default_param}
    mytest_CPU =        {   "CPU_optimized"                     : string_use_cpu + path_benchfdd + string_default_param,
                            "CPU_optimized_segmented"           : string_use_cpu + string_use_segmented + path_benchfdd + string_default_param}
    mytest_CPU_ndm400 = {   "CPU_optimized_ndm400"              : string_use_cpu + path_benchfdd + string_default_param + "-n 400",
                            "CPU_optimized_segmented_ndm400"    : string_use_cpu + string_use_segmented + path_benchfdd + string_default_param + "-n 400"}
    mytest_CPU_ndm800 = {   "CPU_optimized_ndm800"              : string_use_cpu + path_benchfdd + string_default_param + "-n 800",
                            "CPU_optimized_segmented_ndm800"    : string_use_cpu + string_use_segmented + path_benchfdd + string_default_param + "-n 800"}
    mytest_CPU_ndm1200 = {  "CPU_optimized_ndm1200"              : string_use_cpu + path_benchfdd + string_default_param + "-n 1200",
                            "CPU_optimized_segmented_ndm1200"    : string_use_cpu + string_use_segmented + path_benchfdd + string_default_param + "-n 1200"}
    mytest_GPU =        {   "GPU_FDD"                           : string_numactl + path_benchfdd + string_default_param,
                            "GPU_TDD"                           : string_numactl + path_benchdedisp + string_default_param}
    mytest_GPU_ndm400 = {   "GPU_FDD_ndm400"                    : string_numactl + path_benchfdd + string_default_param + "-n 400",
                            "GPU_TDD_ndm400"                    : string_numactl + path_benchdedisp + string_default_param + "-n 400"}
    mytest_GPU_ndm800 = {   "GPU_FDD_ndm800"                    : string_numactl + path_benchfdd + string_default_param + "-n 800",
                            "GPU_TDD_ndm800"                    : string_numactl + path_benchdedisp + string_default_param + "-n 800"}
    mytest_GPU_ndm1200 = {  "GPU_FDD_ndm1200"                    : string_numactl + path_benchfdd + string_default_param + "-n 1200",
                            "GPU_TDD_ndm1200"                    : string_numactl + path_benchdedisp + string_default_param + "-n 1200"}
    mytest_GPU_segmented = {"GPU_FDD_segmented"                 : string_numactl + string_use_segmented + path_benchfdd + string_default_param,
                            "GPU_FDD_segmented_ndm400"          : string_numactl + string_use_segmented + path_benchfdd + string_default_param + "-n 400",
                            "GPU_FDD_segmented_ndm800"          : string_numactl + string_use_segmented + path_benchfdd + string_default_param + "-n 800",
                            "GPU_FDD_segmented_ndm1200"          : string_numactl + string_use_segmented + path_benchfdd + string_default_param + "-n 1200"}

    mytest_CPU_ref_nsamp48 =    {   "CPU_reference_nsamp48"                     : string_use_cpu_ref + path_benchfdd + string_nsamp48,
                                    "CPU_reference_segmented_nsamp48"           : string_use_cpu_ref + string_use_segmented + path_benchfdd + string_nsamp48}
    mytest_CPU_nsamp48 =        {   "CPU_optimized_nsamp48"                     : string_use_cpu + path_benchfdd + string_nsamp48,
                                    "CPU_optimized_segmented_nsamp48"           : string_use_cpu + string_use_segmented + path_benchfdd + string_nsamp48}
    mytest_CPU_ndm400_nsamp48 = {   "CPU_optimized_ndm400_nsamp48"              : string_use_cpu + path_benchfdd + string_nsamp48 + "-n 400",
                                    "CPU_optimized_segmented_ndm400_nsamp48"    : string_use_cpu + string_use_segmented + path_benchfdd + string_nsamp48 + "-n 400"}
    mytest_CPU_ndm800_nsamp48 = {   "CPU_optimized_ndm800_nsamp48"              : string_use_cpu + path_benchfdd + string_nsamp48 + "-n 800",
                                    "CPU_optimized_segmented_ndm800_nsamp48"    : string_use_cpu + string_use_segmented + path_benchfdd + string_nsamp48 + "-n 800"}
    mytest_CPU_ndm1200_nsamp48 = {  "CPU_optimized_ndm1200_nsamp48"             : string_use_cpu + path_benchfdd + string_nsamp48 + "-n 1200",
                                    "CPU_optimized_segmented_ndm1200_nsamp48"   : string_use_cpu + string_use_segmented + path_benchfdd + string_nsamp48 + "-n 1200"}
    mytest_GPU_nsamp48 =        {   "GPU_FDD_nsamp48"                           : string_numactl + path_benchfdd + string_nsamp48,
                                    "GPU_TDD_nsamp48"                           : string_numactl + path_benchdedisp + string_nsamp48}
    mytest_GPU_ndm400_nsamp48 = {   "GPU_FDD_ndm400_nsamp48"                    : string_numactl + path_benchfdd + string_nsamp48 + "-n 400",
                                    "GPU_TDD_ndm400_nsamp48"                    : string_numactl + path_benchdedisp + string_nsamp48 + "-n 400"}
    mytest_GPU_ndm800_nsamp48 = {   "GPU_FDD_ndm800_nsamp48"                    : string_numactl + path_benchfdd + string_nsamp48 + "-n 800",
                                    "GPU_TDD_ndm800_nsamp48"                    : string_numactl + path_benchdedisp + string_nsamp48 + "-n 800"}
    mytest_GPU_ndm1200_nsamp48 = {  "GPU_FDD_ndm1200_nsamp48"                   : string_numactl + path_benchfdd + string_nsamp48 + "-n 1200",
                                    "GPU_TDD_ndm1200_nsamp48"                   : string_numactl + path_benchdedisp + string_nsamp48 + "-n 1200"}
    mytest_GPU_segmented_nsamp48 = {"GPU_FDD_segmented_nsamp48"                 : string_numactl + string_use_segmented + path_benchfdd + string_nsamp48,
                                    "GPU_FDD_segmented_ndm400_nsamp48"          : string_numactl + string_use_segmented + path_benchfdd + string_nsamp48 + "-n 400",
                                    "GPU_FDD_segmented_ndm800_nsamp48"          : string_numactl + string_use_segmented + path_benchfdd + string_nsamp48 + "-n 800",
                                    "GPU_FDD_segmented_ndm1200_nsamp48"          : string_numactl + string_use_segmented + path_benchfdd + string_nsamp48 + "-n 1200"}

    # Select which tests to run
    mytests = mytest_CPU_ref
    mytests.update(mytest_CPU)
    mytests.update(mytest_CPU_ndm400)
    mytests.update(mytest_CPU_ndm800)
    mytests.update(mytest_CPU_ndm1200)
    mytests.update(mytest_GPU)
    mytests.update(mytest_GPU_ndm400)
    mytests.update(mytest_GPU_ndm800)
    mytests.update(mytest_GPU_ndm1200)
    mytests.update(mytest_GPU_segmented)
    #nsamp48
    mytests.update(mytest_CPU_ref_nsamp48)
    mytests.update(mytest_CPU_nsamp48)
    mytests.update(mytest_CPU_ndm400_nsamp48)
    mytests.update(mytest_CPU_ndm800_nsamp48)
    mytests.update(mytest_CPU_ndm1200_nsamp48)
    mytests.update(mytest_GPU_nsamp48)
    mytests.update(mytest_GPU_ndm400_nsamp48)
    mytests.update(mytest_GPU_ndm800_nsamp48)
    mytests.update(mytest_GPU_ndm1200_nsamp48)
    mytests.update(mytest_GPU_segmented_nsamp48)

    #Get start times and format them
    starttime=time.localtime()
    totaltime = time.time()
    print(f'Starting application at {time.strftime("%a %b %d %H:%M:%S %Y",starttime)}')

    # Check paths, create directory, set path variables
    directory_datetime = time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())
    directory = 'bench_results_'+directory_datetime
    try:
        os.mkdir(directory)
    except OSError as error:
        print(error)
    print(f'Created directory {directory} for bechmark run results')

    print(f'Found {len(mytests)} tests to run:')
    print(mytests)

    #Loop over all tests
    for testentry in mytests:
        print(f'### Running test: \"{testentry}\" out of {len(mytests)}')
        print(f'{mytests[testentry]}')

        #Create file to store output
        path = os.path.join(directory,testentry+".txt")
        try:
            f = open(path, "w")
        except OSError:
            print(f'Could not open file: {path}')
            break
        f.write(f'### Benchmark: {testentry}, command {mytests[testentry]} ###\n')

        #Loop over application and get timings, show progress bar
        for i in tqdm.tqdm(range(niterations)):
            #Run the application and capture output
            output = subprocess.getoutput(mytests[testentry])
            #Save to file
            stringTestenty = '###\n'
            stringTestenty += f'### Iteration {i} ###\n'
            stringTestenty += '###\n'
            f.write(stringTestenty + output + '\n')
        f.close()

    #Wrap up
    totaltime = time.time()-totaltime
    print('### Ending application at {} processing took {:.2f} seconds'.format(time.strftime("%a %b %d %H:%M:%S %Y",time.localtime()),totaltime))