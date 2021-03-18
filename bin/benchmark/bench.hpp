/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* Methods for benchmarking with libdedisp, refer to README for more info.
* Adapted from simple test application for libdedisp by Paul Ray (2013)
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <iomanip>

#include <ctime>
#include <random>
#include <functional>

#include <Plan.hpp>

#include "external/Stopwatch.h"
#include <omp.h>
#include <getopt.h>
#include <string.h>

// Debug options
#define WRITE_INPUT_DATA  0
#define WRITE_OUTPUT_DATA 0

// Assume input is a 0 mean float and quantize to an unsigned 8-bit quantity
dedisp_byte bytequant(dedisp_float f)
{
  dedisp_float v = f + 127.5f;
  dedisp_byte r;
  if (v>255.0) {
    r= (dedisp_byte)255;
  } else if (v<0.0f) {
    r= (dedisp_byte)0;
  } else {
    r = (dedisp_byte)roundf(v);
  }
  //printf("ROUND %f, %u\n",f,r);
  return r;
}

// Compute mean and standard deviation of an unsigned 8-bit array
void calc_stats_8bit(dedisp_byte *a, dedisp_size n, dedisp_float *mean, dedisp_float *sigma)
{
  // Use doubles to prevent rounding error
  double sum=0.0, sum2=0.0;
  double mtmp=0.0, vartmp;
  double v;
  dedisp_size i;

  // Use corrected 2-pass algorithm from Numerical Recipes
  sum = 0.0;
  for (i=0;i<n;i++) {
    v = (double)a[i];
    sum += v;
  }
  mtmp = sum/n;

  sum = 0.0;
  sum2 = 0.0;
  for (i=0;i<n;i++) {
    v = (double)a[i];
    sum2 += (v-mtmp)*(v-mtmp);
    sum += v-mtmp;
  }
  vartmp = (sum2-(sum*sum)/n)/(n-1);
  *mean = mtmp;
  *sigma = sqrt(vartmp);

  return;
}

// Compute mean and standard deviation of a float array
void calc_stats_float(dedisp_float *a, dedisp_size n, dedisp_float *mean, dedisp_float *sigma)
{
  // Use doubles to prevent rounding error
  double sum=0.0, sum2=0.0;
  double mtmp=0.0, vartmp;
  double v;
  dedisp_size i;

  // Use corrected 2-pass algorithm from Numerical Recipes
  sum = 0.0;
  for (i=0;i<n;i++) {
    sum += a[i];
  }
  mtmp = sum/n;

  sum = 0.0;
  sum2 = 0.0;
  for (i=0;i<n;i++) {
    v = a[i];
    sum2 += (v-mtmp)*(v-mtmp);
    sum += v-mtmp;
  }
  vartmp = (sum2-(sum*sum)/n)/(n-1);
  *mean = mtmp;
  *sigma = sqrt(vartmp);

  return;
}

void usage(void)
{
  printf("Usage: benchdedisp, benchtdd or benchfdd with -n [nr of DM trials] -s [nr of samples] -c [nr of channels]\n\n");
  printf("No arguments        Default settings\n");
  printf("-n [ntrials]        Number of DM trials\n");
  printf("-s [samples]        Number of samples to generate\n");
  printf("-c [samples]        Number of channels to generate\n");
  printf("-r [DM start(,end)] Alternative Start (and optional end) values of DM range to dedisperse (use end instead of -n)\n");
  printf("-t [Tobs]           Alternative observation time to use [seconds] (use instead of -s)\n");
  printf("-i [niterations]    Number of iterations for the dedipsersion plan and execution\n");
  printf("-q                  Quiet; no verbose information to screen\n");
  printf("-h                  This help\n");
  return;
}

// Parameter struct to configure benchmark
struct BenchParameters
{
  // Default values
  dedisp_float  sampletime_base = 64.0E-6; // Base is 64 microsecond time samples
  dedisp_float  downsamp        = 1.0;
  dedisp_float  dt              = downsamp*sampletime_base;
  dedisp_float  f0              = 1581.0;    // MHz (highest channel!)
  dedisp_float  bw              = 400.0; // MHz
  dedisp_size   nchans          = 1024;
  dedisp_float  df              = -1.0*bw/nchans;   // MHz   (This must be negative!)
  dedisp_float  dm_start        = 0.0;
  dedisp_float  dm_end          = 0.0;
  dedisp_float  dm_step         = 2.0;
  dedisp_size   dm_count       = 100;
  dedisp_size   nsamps          = 120000;
  dedisp_float  Tobs            = 0.0;
  bool          verbose         = true;
  int           niterations     = 1;
};

// Parse input parameters, to be used with benchmarking run method
// return -1 on error
int parseParameters(int argc,char *argv[], BenchParameters & benchParameter)
{
  int arg=0;
  // Decode options
  if (argc>1)
  {
    while ((arg=getopt(argc,argv,"r:s:n:c:t:i:q:h:"))!=-1)
    {
      switch(arg)
      {
        case 'r':
          if (strchr(optarg,',')!=NULL)
            {
              sscanf(optarg,"%f,%f", &benchParameter.dm_start, &benchParameter.dm_end);
              benchParameter.dm_count = 0;
            }
          else
            benchParameter.dm_start=atof(optarg);
        break;

        case 'n':
          benchParameter.dm_count=(dedisp_size) atoi(optarg);
        break;

        case 's':
          benchParameter.nsamps=atoi(optarg);
        break;

        case 'c':
          benchParameter.nchans=atoi(optarg);
        break;

        case 't':
          benchParameter.Tobs=atoi(optarg);
        break;

        case 'i':
          benchParameter.niterations=atoi(optarg);
        break;

        case 'q':
          benchParameter.verbose=0;
        break;

        case 'h':
          usage();
          return -1;
        break;

        default:
          usage();
        return -1;
      }
    } // while
  }
  else
  {
    usage();
    printf("Using default settings\n");
  }

  // Set end DM based on number of DM's specified
  if (benchParameter.dm_count!=0 && benchParameter.dm_end==0.0)
  {
    benchParameter.dm_end = benchParameter.dm_start + benchParameter.dm_count * benchParameter.dm_step;
  }
  else
  {
    fprintf(stderr,"Error parsing DM range. Provide number of dms or start and end of a range.\n");
    return -1;
  }

  // re calculate some parameters for if there were changes:
  benchParameter.dt = benchParameter.downsamp*benchParameter.sampletime_base; // we might make sampletime_base configurable later
  benchParameter.df = -1.0*benchParameter.bw/benchParameter.nchans; // changes with nchans

  // samples versus time
  if (benchParameter.Tobs != 0) // if time was defined then define number of samples based on time
  {
    benchParameter.nsamps = benchParameter.Tobs / benchParameter.dt;
  }
  else // define time based on number of samples
  {
    benchParameter.Tobs = benchParameter.nsamps * benchParameter.dt;
  }

  return 0;
}

// run method for benchmarking function for a dedisp, tdd or fdd Plan
template<typename PlanType>
int run(BenchParameters & benchParameter)
{
  int          device_idx  = 0;

  dedisp_float downsamp    = benchParameter.downsamp;
  dedisp_float Tobs        = benchParameter.Tobs;
  dedisp_float dt          = benchParameter.dt;
  dedisp_float f0          = benchParameter.f0;
  dedisp_size  nchans      = benchParameter.nchans;
  dedisp_float df          = benchParameter.df;
  dedisp_size  nsamps      = benchParameter.nsamps;

  dedisp_float dm_start    = benchParameter.dm_start;
  dedisp_float dm_end      = benchParameter.dm_end;
  dedisp_float pulse_width = 4.0;   // ms
  dedisp_float dm_tol      = 1.25;
  dedisp_size  in_nbits    = 8;
  dedisp_size  out_nbits   = 32;  // DON'T CHANGE THIS FROM 32, since that signals it to use floats

  dedisp_size  dm_count    = benchParameter.dm_count;
  dedisp_float dm_step     = benchParameter.dm_step;
  dedisp_size  max_delay;
  dedisp_size  nsamps_computed;

  dedisp_float *dmlist;

  std::unique_ptr<Stopwatch> tbench(Stopwatch::create());
  std::unique_ptr<Stopwatch> tinit(Stopwatch::create());
  std::unique_ptr<Stopwatch> tplan(Stopwatch::create());
  std::unique_ptr<Stopwatch> texecute(Stopwatch::create());
  tbench->Start();
  tinit->Start();

  printf("Starting benchmark for %d iterations \n", benchParameter.niterations);

  auto maxBufferSize = std::max(nsamps * nchans * (in_nbits/8) * sizeof(dedisp_float), nsamps * dm_count * (out_nbits/8) * sizeof(dedisp_byte));

  if(benchParameter.verbose)
  {
    printf("------------------------ INPUT AND OUTPUT DATA ---------------------------\n");
    printf("Frequency of highest chanel [MHz]            : %.4f\n",f0);
    printf("NCHANS (Channel Width [MHz])                 : %lu (%f)\n",nchans,df);
    printf("Sample time (after downsampling by %.0f [us])   : %f\n",downsamp,dt/1E-6);
    printf("Observation duration [s]                     : %f (%lu samples)\n",Tobs,nsamps);
    printf("Input data                                   : %2lu bits \n",in_nbits);
    printf("Output data                                  : %2lu bits \n",out_nbits);
    printf("Input data array size                        : %lu MB\n",(nsamps*nchans*(in_nbits/8)*sizeof(dedisp_float))/(1<<20));
    printf("Output (max) data array size                 : %lu MB\n",(nsamps*dm_count*(out_nbits/8)*sizeof(dedisp_byte))/(1<<20));
    printf("Shared input/output dummy buffer size        : %lu MB, %lu GB\n",maxBufferSize/(1<<20), maxBufferSize/(1<<30));
    printf("\n");
  }

  /* Allocate a shared buffer for input and output, don't care about the contents.
  This saves benchmarking initialization time but does not impact implementation benchmarking time,
  because the implementation performance is not data dependent. */

  auto dummyData = (dedisp_float *) malloc(maxBufferSize);
  if (dummyData == NULL) {
    printf("\nERROR: Failed to allocate shared input/output dummy buffer array\n");
    return -1;
  }

  /* Both input and output point to the same dummy memory location, but with different pointer types.
  This saves us a lot of RAM utiliztion. Read and write at the same time should not be an issue. */
  dedisp_byte *input  = (dedisp_byte *) dummyData;
  dedisp_float *output = (dedisp_float *) dummyData;

  if(benchParameter.verbose)
  {
    printf("\n");
    printf("----------------------------- INJECTED SIGNAL  ----------------------------\n");
    printf("Uninitialized dummy data \n");
  }
  tinit->Pause();

  for(int iPlan = 0; iPlan < benchParameter.niterations; iPlan++)
  {
    tplan->Start();

    printf("\n");
    printf("------------------------- ITERATION %d out of %d  -------------------------\n", iPlan+1, benchParameter.niterations);

    if(benchParameter.verbose) printf("Create plan and init GPU\n");
    // Create a dedispersion plan
    PlanType plan(nchans, dt, f0, df, device_idx);

    if(benchParameter.verbose) printf("Gen DM list\n");
    // Generate a list of dispersion measures for the plan
    if (dm_count==0)
    {
      if (benchParameter.verbose) printf("Generating optimal DM trials\n");
      plan.generate_dm_list(dm_start,dm_end,pulse_width,dm_tol);
    }
    else
    { // Generate a list of dispersion measures for the plan
      if (benchParameter.verbose) printf("Generating linear DM trials\n");
      dmlist=(dedisp_float *) calloc(sizeof(dedisp_float),dm_count);
      for (dedisp_size i=0;i<dm_count;i++)
      {
        dmlist[i]=(dedisp_float) dm_start+dm_step*i;
      }
      plan.set_dm_list(dmlist,dm_count);
    }

    // Find the parameters that determine the output size
    dm_count = plan.get_dm_count();
    max_delay = plan.get_max_delay();
    nsamps_computed = nsamps - max_delay;
    dmlist=(float *)plan.get_dm_list();

    if(benchParameter.verbose)
    {
      printf("\n");
      printf("----------------------------- DM COMPUTATIONS  ----------------------------\n");
      printf("Computing %lu DMs from %f to %f pc/cm^3\n",dm_count,dmlist[0],dmlist[dm_count-1]);
      printf("Max DM delay is %lu samples (%.f seconds)\n",max_delay,max_delay*dt);
      printf("Computing %lu out of %lu total samples (%.2f%% efficiency)\n",nsamps_computed,nsamps,100.0*(dedisp_float)nsamps_computed/nsamps);
      printf("\n");
    }

    tplan->Pause();
    texecute->Start();

    printf("\n");
    printf("--------------------------- PERFORM DEDISPERSION  -------------------------\n");
#ifndef DEDISP_BENCHMARK
    printf("\n>>> WARNING: not showing timing output because define DEDISP_BENCHMARK is not set\n\n");
#endif
    // Compute the dedispersion transform on the GPU
    plan.execute(nsamps,
        input, in_nbits,
        (dedisp_byte *) output, out_nbits);
    texecute->Pause();

    // Print timings
    if(benchParameter.verbose)
    {
      tbench->Pause();
      printf("\n");
      printf("------------- BENCHMARK TIMES (accumulated for %d iteration(s)) -------------\n", iPlan+1);
      std::cout << "Benchmark total time:      " << tbench->ToString() << " sec." << std::endl;
      std::cout << "Benchmark init time:       " << tinit->ToString() << " sec." << std::endl;
      std::cout << "Benchmark plan time:       " << tplan->ToString() << " sec." << std::endl;
      std::cout << "Benchmark execute time:    " << texecute->ToString() << " sec." << std::endl;
      tbench->Start();
    }
  }// end for niterations
  tbench->Pause();

  // Print timings
  if(!benchParameter.verbose)
  {
    printf("\n");
    printf("------------- BENCHMARK TIMES (accumulated for %d iteration(s)) -------------\n", benchParameter.niterations);
    std::cout << "Benchmark total time:      " << tbench->ToString() << " sec." << std::endl;
    std::cout << "Benchmark init time:       " << tinit->ToString() << " sec." << std::endl;
    std::cout << "Benchmark plan time:       " << tplan->ToString() << " sec." << std::endl;
    std::cout << "Benchmark execute time:    " << texecute->ToString() << " sec." << std::endl;
  }

  // Clean up
  free(dummyData);
  printf("Benchmark finished\n");
  return 0;
}