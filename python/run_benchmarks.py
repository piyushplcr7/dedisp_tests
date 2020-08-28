#!/usr/bin/env python3
#Script to run muliple consecutive benchmarks and store output

import subprocess
import re
import numpy as np
import time
import tqdm #progress bar
import sys
import os

if __name__ == "__main__":
    #Run application and capture output

    #How many times to repeat each test
    niterations=5

    #Which tests to run
    # mytests is a dictionay of benchmark names and executable commands
    # Use full paths in the commands
    # No spaces in mytest name!
    # For reference
    string_numactl          =   "numactl --cpunodebind=0 "
    path_benchfdd           =   "/home/vlugt/bin/dedisp/bin/benchfdd "
    path_benchdedisp        =   "/home/vlugt/bin/dedisp/bin/benchfdd "
    string_use_cpu          =   "USE_CPU=1 "
    string_use_cpu_ref      =   "USE_CPU=1 USE_REFERENCE=1 "
    string_use_segmented    =   "USE_SEGMENTED=1 "
    string_nchan            =   "-s 240000 "
    string_default_param    =   string_nchan
    mytests_stub = {    "Test1": "/home/vlugt/dedisp/python/print.sh 1",
                        "Test2": "/home/vlugt/dedisp/python/print.sh 2",
                        "Test3": "/home/vlugt/dedisp/python/print.sh 3"}
    mytest_CPU_ref =    {   "CPU_reference"                     : string_use_cpu_ref + path_benchfdd + string_default_param,
                            "CPU_reference_segmented"           : string_use_cpu_ref + string_use_segmented + path_benchfdd + string_default_param}
    mytest_CPU =        {   "CPU_optimized"                     : string_use_cpu + path_benchfdd + string_default_param,
                            "CPU_optimized_segmented"           : string_use_cpu + string_use_segmented + path_benchfdd + string_default_param}
    mytest_CPU_ndm400 = {   "CPU_optimized_ndm400"              : string_use_cpu + path_benchfdd + string_default_param + "-n 400",
                            "CPU_optimized_segmented_ndm400"    : string_use_cpu + string_use_segmented + path_benchfdd + string_default_param + "-n 400"}
    mytest_CPU_ndm800 = {   "CPU_optimized_ndm800"              : string_use_cpu + path_benchfdd + string_default_param + "-n 800",
                            "CPU_optimized_segmented_ndm800"    : string_use_cpu + string_use_segmented + path_benchfdd + string_default_param + "-n 800"}
    mytest_GPU =        {   "GPU_FDD"                           : string_numactl + path_benchfdd + string_default_param,
                            "GPU_TDD"                           : string_numactl + path_benchdedisp + string_default_param}
    mytest_GPU_ndm400 = {   "GPU_FDD_ndm400"                    : string_numactl + path_benchfdd + string_default_param + "-n 400",
                            "GPU_TDD_ndm400"                    : string_numactl + path_benchdedisp + string_default_param + "-n 400"}
    mytest_GPU_ndm800 = {   "GPU_FDD_ndm800"                    : string_numactl + path_benchfdd + string_default_param + "-n 800",
                            "GPU_TDD_ndm800"                    : string_numactl + path_benchdedisp + string_default_param + "-n 800"}

    # Select which tests to run
    mytests = mytest_CPU_ref
    mytests.update(mytest_CPU)
    mytests.update(mytest_CPU_ndm400)
    mytests.update(mytest_CPU_ndm800)
    mytests.update(mytest_GPU)
    mytests.update(mytest_GPU_ndm400)
    mytests.update(mytest_GPU_ndm800)

    # mytests = ["numactl --cpunodebind=0 /home/vlugt/bin/dedisp/bin/fdd_fil -f /var/scratch/vlugt/fdd/data/pks_frb110220.fil -r 930 -s 1 -n 256 -o pks_frb110220",\
    #             "numactl --cpunodebind=0 /home/vlugt/bin/dedisp/bin/dedisp_fil -f /var/scratch/vlugt/fdd/data/pks_frb110220.fil -r 930 -s 1 -n 256 -o pks_frb110220",\
    #             "numactl --cpunodebind=0 /home/vlugt/bin/dedisp/bin/fdd_fil -f /var/scratch/vlugt/fdd/data/pks_frb110220.fil -r 674 -s 1 -n 512 -o pks_frb110220",\
    #             "numactl --cpunodebind=0 /home/vlugt/bin/dedisp/bin/dedisp_fil -f /var/scratch/vlugt/fdd/data/pks_frb110220.fil -r 674 -s 1 -n 512 -o pks_frb110220"]
    # mytests = ["numactl --cpunodebind=0 /home/vlugt/bin/dedisp/bin/dedisp_fil -f /var/scratch/vlugt/fdd/data/pks_frb110220.fil -r 930 -s 1 -n 256 -o pks_frb110220"]
    #["numactl --cpunodebind=0 /home/vlugt/bin/dedisp/bin/fdd_fil -f /var/scratch/vlugt/fdd/data/pks_frb110220.fil -r 930 -s 1 -n 256 -o pks_frb110220"]
    # mytests = ["/home/vlugt/bin/dedisp/bin/benchfdd"]

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
            #Run the application
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