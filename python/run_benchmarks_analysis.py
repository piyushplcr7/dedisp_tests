#!/usr/bin/env python3
#Script to analyze muliple consecutive benchmarks
# Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
# SPDX-License-Identifier: GPL-3.0-or-later
# Script to analyze data that has been created with run_benchmarks.py.
# Output is printed to stdout,
# redirect it to a file if you want to store the analysis output

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
    parser = argparse.ArgumentParser(description='Analyze muliple consecutive benchmarks')
    parser.add_argument('inputDirectory',
                    help='The directory with the benchmark results')
    parser.add_argument('--quiet',
                    help='Suppress verbose output')
    return parser

# Get runtimes from strings
def get_timing (stdout, timing):
    # Get all strings with timing results
    items=re.findall("^.*"+timing+".*$",stdout,re.MULTILINE)
    # print(items)

    # Look for floating point number in string
    # get just the number matching xx.yy and return it as array
    numeric_const_pattern = '[+-]?[0-9]+\.[0-9]+'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    results = np.zeros(len(items)) # init
    idx=0
    for item in items:
        timestr=rx.findall(item)
        if len(timestr) != 1: # expecting only one timing result per line
                print('Warning: looking for float number in timing \"{}\" found {} matches instead of 1, working on item:'.format(timing, len(timestr)))
                print(item)
                print(items)
                return np.nan # return NaN and continue
        #and convert it
        results[idx] = float(timestr[0])
        idx = idx+1
    return results

# Formatted print a dict of dicts
def prettyPrintDict(dictToPrint):
    for key_outer in dictToPrint:
        printString = f'{key_outer} '
        tempDict = dictToPrint[key_outer]
        for key_inner in tempDict:
            printString += f'{key_inner} {tempDict[key_inner]} '
        print(printString)
    return

if __name__ == "__main__":
    #Run application and capture output

    # Assumptions on te data:
    # Strings printed to stdout, with a prefix <mytest> and terminated by a line ending
    # Matching one (and only one) of the strings of mytiming
    # and then a non zero floating point number of the format x.y
    # where x and y should be at least, but can be more than, one digit

    # What data to collect (same for all tests), returns NaN when not available for the test
    mytimings = (   "Initialization time : ",
                    "Preprocessing time  : ",
                    "Dedispersion time   : ",
                    "Postprocessing time : ",
                    "Input memcpy time   : ",
                    "Output memcpy time  : ",
                    "Runtime             : ",
                    "GPU execution time  : ",
                    "Total time          : ")
    # Init dicts
    allmydata = {}
    meanTotals = {}
    allmymeandata = {}

    # Get application start times and format them
    starttime=time.localtime()
    totaltime = time.time()
    print('Starting application at {}'.format(time.strftime("%a %b %d %H:%M:%S %Y",starttime)))

    # Get input parameters
    argParser = create_arg_parser()
    parsedArgs = argParser.parse_args(sys.argv[1:])
    verbose = not(parsedArgs.quiet)

    # Set main directory
    if os.path.exists(parsedArgs.inputDirectory):
        inputDirectory = parsedArgs.inputDirectory
        # Count files in te directory
        fileList = os.listdir(inputDirectory)
        nfiles = len(fileList)
        print(f'Found {nfiles} files in directory {inputDirectory}')
        print(fileList)
    else: # raise exception
        raise("Directory" + parsedArgs.inputDirectory + "does not exist")

    # Iterate over all files in the directory, file name is the test name
    for i in range(nfiles):
        # Init
        mydata = {} # Dict for timing name + timing values
        # Open te file, get test name (based on file name)
        fileName = fileList[i]
        filePath = os.path.join(inputDirectory, fileName)
        testName = str.split(fileName,'.')[0]
        if(verbose): print(f'\n###\n### Processing benchmark for {testName}\n###\n')
        f = open(filePath)
        benchmarkResults = f.read()

        # Get my timings, return results for multiple instances (iterations) of the timing
        for timing in mytimings:
            mydata[timing]=get_timing(benchmarkResults, timing) # Returns an array with values
        allmydata[testName]=mydata # Dict with for each test name a dict with test names and values
        f.close()

        if(verbose):
            # Process data and display
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


    # Create datastructures with only mean times per test
    if(verbose):
        print("### Re-formatting data to mean values:")
        for testName in allmydata:
            # Create Dict with test name with dict of timing name and mean value
            print(f'\n### Benchmark \n{testName}')
            mymeandata = {}
            for timing in mytimings:
                countnan=np.count_nonzero(np.isnan(allmydata[testName][timing]))
                if((len(allmydata[testName][timing])>0) and not(countnan)): # if it contains data
                    mymeandata[timing] = np.nanmean(allmydata[testName][timing])
                    print('{:30} {:.4f}'.format(timing, mymeandata[timing]))
                else:
                    mymeandata[timing] = np.nan
                    print('{:30} Not available'.format(timing))
            allmymeandata[testName]=mymeandata
        #prettyPrintDict(allmymeandata)

    # Summarized overview of results
    print('\n')
    print(f'### Summary of mean values:')
    summaryFormatString = "{:45}".format("Test name")
    summaryFormatString += ":"
    summaryFormatString += "{:45}".format("Sort by : : : : ")  # For easy sorting in spreadsheet
    summaryFormatString += ": "
    for timing in mytimings:
        summaryFormatString += "{:20}".format(timing)
    print(summaryFormatString)
    for testName in allmymeandata:
        summaryResultString = "{:45}".format(testName)
        # For easy sorting in spreadsheet:
        temptestName = testName.replace('_',':')
        temptestName = temptestName.replace('nchan','')
        temptestName = temptestName.replace('nsamp','')
        temptestName = temptestName.replace('ndm','')
        summaryResultString += ":{:45}".format(temptestName)
        # Nice to have (ToDo):
        # Implement a way to sort and cluser data, cluser by: 1) implementation, 2) nsamp, 3) ndm
        # e.g. for testName in sort(allmymeandata) ... but then for a dict of dicts on elements of the key
        # e.g. testName.split('_') # returns array e.g. ['GPU', 'tdd', 'nsamp448', 'ndm512']
        # For now just as csv (':') for easy import, sorting and analysis in spreadsheet
        for timing in mytimings:
            timingstr = "{:4.6f}".format(allmymeandata[testName][timing])
            summaryResultString += ": {:20}".format(timingstr)
        print(summaryResultString)

    #Wrap up
    totaltime = time.time()-totaltime
    print('### Ending application at {} processing took {:.2f} seconds'.format(time.strftime("%a %b %d %H:%M:%S %Y",time.localtime()),totaltime))