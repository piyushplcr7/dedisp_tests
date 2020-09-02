#!/usr/bin/env python3
#Script to analyze muliple consecutive benchmarks

import subprocess
import re
import numpy as np
import time
import tqdm #progress bar
import sys
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt

def create_arg_parser():
    # Creates and returns the ArgumentParser object

    parser = argparse.ArgumentParser(description='Analyze muliple consecutive benchmarks')
    parser.add_argument('inputDirectory',
                    help='The directory with the benchmark results')
    parser.add_argument('--quiet',
                    help='Suppress verbose output')
    parser.add_argument('--showPlot',
                    help='If specified the plot is generated and showed to screen')
    parser.add_argument('--savePlot',
                    help='If specified the plot is generated and saved to the specified file path')
    return parser

def get_timing (stdout, timing):
    #Look for time string
    items=re.findall("^.*"+timing+".*$",stdout,re.MULTILINE)
    # print(items)

    #Look for floating point number in string
    # https://stackoverflow.com/a/4703508
    # all numbers:
    # numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?' #returns array
    # all floats, returns tuple with both the float and only the decimal part
    # numeric_const_pattern = '[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?'
    # get just number matching xx.yy
    numeric_const_pattern = '[+-]?[0-9]+\.[0-9]+' #returns array
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    # init
    results = np.zeros(len(items))
    idx=0
    for item in items:
        timestr=rx.findall(item)
        if len(timestr) != 1:
                print('Warning: looking for float number in timing \"{}\" found {} matches instead of 1, working on item:'.format(timing, len(timestr)))
                print(item)
                print(items)
                return np.nan
        #and convert it
        results[idx] = float(timestr[0])
        idx = idx+1
    return results

if __name__ == "__main__":
    #Run application and capture output

    #Assumptions on te data:
    # Strings printed to stdout, with first a prefix <mytest>
    # and then a non zero floating point number of the format x.y
    # where x and y should be at least, but can be more than, one digit

    #What data to collect (same for all tests)
    mytimings = (   "Initialization time : ",
                    "Preprocessing time  : ",
                    "Dedispersion time   : ",
                    "Postprocessing time : ",
                    "Input memcpy time   : ",
                    "Output memcpy time  : ",
                    "Total time          : ")
    #Init dicts
    allmydata = {}
    meanTotals = {}
    allmymeandata = {}

    #Get application start times and format them
    starttime=time.localtime()
    totaltime = time.time()
    print('Starting application at {}'.format(time.strftime("%a %b %d %H:%M:%S %Y",starttime)))

    #Get input parameters
    argParser = create_arg_parser()
    parsedArgs = argParser.parse_args(sys.argv[1:])
    verbose = not(parsedArgs.quiet)

    #Set main directory
    if os.path.exists(parsedArgs.inputDirectory):
        inputDirectory = parsedArgs.inputDirectory
        #Count files in te directory
        fileList = os.listdir(inputDirectory)
        nfiles = len(fileList)
        print(f'Found {nfiles} files in directory {inputDirectory}')
        print(fileList)
    else: #raise exception
        raise("Directory" + parsedArgs.inputDirectory + "does not exist")

    # # test
    # fileName = fileList[1]
    # filePath = os.path.join(inputDirectory, fileName)
    # testName = str.split(fileName,'.')[0]
    # f = open(filePath)
    # benchmarkResults = f.read()

    # #Get my timings
    # for timing in mytimings:
    #      temp=get_timing(benchmarkResults,timing)
    #      print('Found:')
    #      print(temp)
    #     #Returns np.nan on error
    # f.close()
    # raise("Breakpoint")

    # Iterate over all files in the directory, file name is the test name
    for i in range(nfiles):
        #Init
        mydata = {} #Dict for timing name + timing values
        #Open te file, get test name (based on file name)
        fileName = fileList[i]
        filePath = os.path.join(inputDirectory, fileName)
        testName = str.split(fileName,'.')[0]
        if(verbose): print(f'\n###\n### Processing benchmark for {testName}\n###\n')
        f = open(filePath)
        benchmarkResults = f.read()

        #Get my timings, return results for multiple instances (iterations) of the timing
        for timing in mytimings:
            mydata[timing]=get_timing(benchmarkResults, timing) #Returns an array with values
        allmydata[testName]=mydata #Dict with for each test name a dict with test names and values
        f.close()

        if(verbose):
            #Process data and display
            print(f'### Sanity check:')
            for timing in mytimings:
                countnonzero=np.count_nonzero(mydata[timing])
                countnan=np.count_nonzero(np.isnan(mydata[timing]))
                print('Timing \"{:30}\" has {} valid entries, {} nonzero entries and {} NaN entries'.format(timing, countnonzero-countnan, countnonzero, countnan))

            print('### Statistics:')
            for timing in mytimings:
                countnonzero=np.count_nonzero(mydata[timing])
                countnan=np.count_nonzero(np.isnan(mydata[timing]))
                if(not(countnan) and countnonzero): # if it contains data
                    print('{:30}, min: {:.4f}, max: {:.4f}, mean: {:.4f}, std: {:.4f}, seconds'.format(timing, np.nanmin(mydata[timing]), np.nanmax(mydata[timing]), np.nanmean(mydata[timing]), np.nanstd(mydata[timing])))

            print('### Statistics means:')
            for timing in mytimings:
                countnonzero=np.count_nonzero(mydata[timing])
                countnan=np.count_nonzero(np.isnan(mydata[timing]))
                if(not(countnan) and countnonzero): # if it contains data
                    print('{:30}: {:.4f}'.format(timing, np.nanmean(mydata[timing])))


    #Create datastructures with only mean times per test
    print("### Re-formatting data to mean values:")
    for testName in allmydata:
        #Create Dict with test name with dict of timing name and mean value
        print(f'\n### Benchmark \n{testName}')
        mymeandata = {}
        for timing in mytimings:
            countnonzero=np.count_nonzero(allmydata[testName][timing])
            countnan=np.count_nonzero(np.isnan(allmydata[testName][timing]))
            if(not(countnan) and countnonzero): # if it contains data
                mymeandata[timing] = np.nanmean(allmydata[testName][timing])
                print('{:30} {:.4f}'.format(timing, mymeandata[timing]))
            else: print('{:30} Contains errors'.format(timing))
        #if(verbose): print(f'Re-formatted data for {testName} {timing} to a structure with shape {len(mymeandata)}')
        #print(mymeandata)
        allmymeandata[testName]=mymeandata

    #Create datastructures with only mean times per test
    tidx=mytimings[len(mytimings)-1]
    print('\n')
    print(f'### Summary of mean totals for \"{tidx}\":')
    for testName in allmydata:
        #Create Dict with total mean times
        countnonzero=np.count_nonzero(allmydata[testName][tidx])
        countnan=np.count_nonzero(np.isnan(allmydata[testName][tidx]))
        if(not(countnan) and countnonzero): # if it contains data
            #print(allmydata[testName])
            #print(allmydata[testName][tidx])
            meanTotals[testName] = np.nanmean(allmydata[testName][tidx])
            print('{:30}: {:.4f}'.format(testName, meanTotals[testName]))
        else: print('{:30}: Contains errors'.format(testName))
    #print(meanTotals)
       # for timing in mytimings:

    #Bar plot of mean totals:
    # keys = meanTotals.keys()
    # values = meanTotals.values()
    # plt.bar(keys, values)
    # plt.show()

    # Plot ?
    # if parsedArgs.savePlot:
    #     plt.savefig(parsedArgs.savePlot)
    #     print("Saved plot to " + parsedArgs.savePlot)
    # else:
    #     plt.show()

    #Wrap up
    totaltime = time.time()-totaltime
    print('### Ending application at {} processing took {:.2f} seconds'.format(time.strftime("%a %b %d %H:%M:%S %Y",time.localtime()),totaltime))