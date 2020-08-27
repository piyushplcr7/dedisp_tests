/*
  Simple test application for libdedisp
  By Paul Ray (2013)
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
  printf("Usage: benchfdd or benchdedisp with -n [nr of DM trials] -s [nr of samples] -c [nr of channels]\n\n");
  printf("No arguments        Default settings\n");
  printf("-n [ntrails]        Number of DM trails.\n");
  printf("-s [samples]        Number of samples to generate\n");
  printf("-c [samples]        Number of channels to generate\n");
  printf("-r [DM start(,end)] Alternative Start (and optional end) values of DM range to dedisperse\n");
  printf("-t [Tobs]           Alternative observation time to use, use instead of -S\n");
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
};

int parseParameters(int argc,char *argv[], BenchParameters & benchParameter)
{
  int arg=0;
  // Decode options
  if (argc>1)
  {
    while ((arg=getopt(argc,argv,"r:s:n:c:t:q:h:"))!=-1)
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
  if (benchParameter.Tobs != 0) // it time was defined then define number of samples based on time
  {
    benchParameter.nsamps = benchParameter.Tobs / benchParameter.dt;
  }
  else // define time based on number of samples
  {
    benchParameter.Tobs = benchParameter.nsamps * benchParameter.dt;
  }

  return 0;
}

template<typename PlanType>
int run(BenchParameters & benchParameter)
{
  int          device_idx  = 0;

  dedisp_float downsamp    = benchParameter.downsamp;
  dedisp_float Tobs        = benchParameter.Tobs;
  dedisp_float dt          = benchParameter.dt;
  dedisp_float f0          = benchParameter.f0;
  dedisp_float bw          = benchParameter.bw;
  dedisp_size  nchans      = benchParameter.nchans;
  dedisp_float df          = benchParameter.df;
  dedisp_size  nsamps      = benchParameter.nsamps;

  dedisp_float datarms     = 25.0;
  dedisp_float sigDM = 41.159;
  dedisp_float sigT = 3.14159; // seconds into time series (at f0)
  dedisp_float sigamp = 25.0; // amplitude of signal

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
  dedisp_byte *input  = 0;
  dedisp_float *output = 0;

  unsigned int i,nc,ns,nd;
  dedisp_float *dmlist;
  //const dedisp_size *dt_factors;
  dedisp_float *delay_s;

  dedisp_float *rawdata;

  std::unique_ptr<Stopwatch> tbench(Stopwatch::create());
  std::unique_ptr<Stopwatch> tinit(Stopwatch::create());
  std::unique_ptr<Stopwatch> tplan(Stopwatch::create());
  std::unique_ptr<Stopwatch> texecute(Stopwatch::create());
  std::unique_ptr<Stopwatch> tcheck(Stopwatch::create());
  tbench->Start();
  tinit->Start();

  printf("Starting benchmark\n");
  if(benchParameter.verbose)
  {
    printf("----------------------------- INPUT DATA ---------------------------------\n");
    printf("Frequency of highest chanel [MHz]            : %.4f\n",f0);
    printf("Bandwidth (MHz)                              : %.2f\n",bw);
    printf("NCHANS (Channel Width [MHz])                 : %lu (%f)\n",nchans,df);
    printf("Sample time (after downsampling by %.0f [us])    : %f\n",downsamp,dt/1E-6);
    printf("Observation duration [s]                     : %f (%lu samples)\n",Tobs,nsamps);
    printf("Data RMS (%2lu bit input data)                 : %f\n",in_nbits,datarms);
    printf("Input data array size                        : %lu MB\n",(nsamps*nchans*sizeof(float))/(1<<20));
    printf("\n");
  }

  /* Initialize random number generator */
  auto random = std::bind(std::normal_distribution<float>(0, 1),
                          std::mt19937(0));

  /* First build 2-D array of floats with our signal in it */
  rawdata = (dedisp_float *) malloc(nsamps*nchans*sizeof(dedisp_float));

  //#pragma omp parallel for
  for (ns=0; ns<nsamps; ns++) {
    for (nc=0; nc<nchans; nc++) {
      rawdata[ns*nchans+nc] = datarms*random();
    }
  }

  /* Now embed a dispersed pulse signal in it */
  delay_s = (dedisp_float *) malloc(nchans*sizeof(dedisp_float));
  #pragma omp parallel for
  for (nc=0; nc<nchans; nc++) {
    dedisp_float a = 1.f/(f0+nc*df);
    dedisp_float b = 1.f/f0;
    delay_s[nc] = sigDM*4.15e3 * (a*a - b*b);
  }
  if(benchParameter.verbose) printf("Embedding signal\n");
  #pragma omp parallel for
  for (nc=0; nc<nchans; nc++) {
    ns = (int)((sigT + delay_s[nc])/dt);
    if (ns > nsamps) {
      printf("ns too big %u\n",ns);
      exit(1);
    }
    rawdata[ns*nchans + nc] += sigamp;
  }

  if(benchParameter.verbose)
  {
    printf("\n");
    printf("----------------------------- INJECTED SIGNAL  ----------------------------\n");
    printf("Pulse time at f0 (s)                      : %.6f (sample %lu)\n",sigT,(dedisp_size)(sigT/dt));
    printf("Pulse DM (pc/cm^3)                        : %f \n",sigDM);
    printf("Signal Delays : %f, %f, %f ... %f\n",delay_s[0],delay_s[1],delay_s[2],delay_s[nchans-1]);
    /*
      input is a pointer to an array containing a time series of length
      nsamps for each frequency channel in plan. The data must be in
      time-major order, i.e., frequency is the fastest-changing
      dimension, time the slowest. There must be no padding between
      consecutive frequency channels.
    */

    dedisp_float raw_mean, raw_sigma;
    calc_stats_float(rawdata, nsamps*nchans, &raw_mean, &raw_sigma);
    printf("Rawdata Mean (includes signal)    : %f\n",raw_mean);
    printf("Rawdata StdDev (includes signal)  : %f\n",raw_sigma);
    printf("Pulse S/N (per frequency channel) : %f\n",sigamp/datarms);
  }
  tinit->Pause();
  tplan->Start();

  input = (dedisp_byte *) malloc(nsamps * nchans * (in_nbits/8));

  if(benchParameter.verbose) printf("Quantizing array\n");
  /* Now fill array by quantizing rawdata */
  for (ns=0; ns<nsamps; ns++) {
    for (nc=0; nc<nchans; nc++) {
      input[ns*nchans+nc] = bytequant(rawdata[ns*nchans+nc]);
    }
  }

  if(benchParameter.verbose)
  {
    dedisp_float in_mean, in_sigma;
    calc_stats_8bit(input, nsamps*nchans, &in_mean, &in_sigma);

    printf("Quantized data Mean (includes signal)    : %f\n",in_mean);
    printf("Quantized data StdDev (includes signal)  : %f\n",in_sigma);
    printf("\n");
  }

  if(benchParameter.verbose) printf("Create plan\n");
  // Create a dedispersion plan
  PlanType plan(nchans, dt, f0, df);

  if(benchParameter.verbose) printf("Init GPU\n");
  // Initialise the GPU
  plan.set_device(device_idx);

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
    for (i=0;i<dm_count;i++)
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
  //dt_factors = plan.get_dt_factors(plan);

  if(benchParameter.verbose)
  {
    printf("\n");
    printf("----------------------------- DM COMPUTATIONS  ----------------------------\n");
    printf("Computing %lu DMs from %f to %f pc/cm^3\n",dm_count,dmlist[0],dmlist[dm_count-1]);
    printf("Max DM delay is %lu samples (%.f seconds)\n",max_delay,max_delay*dt);
    printf("Computing %lu out of %lu total samples (%.2f%% efficiency)\n",nsamps_computed,nsamps,100.0*(dedisp_float)nsamps_computed/nsamps);
    printf("Output data array size : %lu MB\n",(dm_count*nsamps_computed*(out_nbits/8))/(1<<20));
    printf("\n");
  }

  // Allocate space for the output data
  output = (dedisp_float *) malloc(nsamps_computed * dm_count * out_nbits/8);
  if (output == NULL) {
    printf("\nERROR: Failed to allocate output array\n");
    return -1;
  }

  tplan->Pause();
  texecute->Start();

  printf("\n");
  printf("--------------------------- PERFORM DEDISPERSION  -------------------------\n");
  // Compute the dedispersion transform on the GPU
  plan.execute(nsamps,
			 input, in_nbits,
			 (dedisp_byte *)output, out_nbits);
  texecute->Pause();
  printf("\n");
  printf("------------------------------ CHECK RESULT  ------------------------------\n");
  // Look for significant peaks
  tcheck->Start();
  dedisp_float out_mean, out_sigma;
  calc_stats_float(output, nsamps_computed*dm_count, &out_mean, &out_sigma);

  printf("Output RMS                               : %f\n",out_mean);
  printf("Output StdDev                            : %f\n",out_sigma);

  if(benchParameter.verbose)
  {
  i=0;
    for (nd=0; nd<dm_count; nd++) {
      for (ns=0; ns<nsamps_computed; ns++) {
        dedisp_size idx = nd*nsamps_computed+ns;
        dedisp_float val = output[idx];
        if (val-out_mean > 6.0*out_sigma) {
    printf("DM trial %u (%.3f pc/cm^3), Samp %u (%.6f s): %f (%.2f sigma)\n",nd,dmlist[nd],ns,ns*dt,val,(val-out_mean)/out_sigma);
    i++;
    if (i>100)
      break;
        }
      }
      if (i>100)
        break;
    }
  }

  #if WRITE_INPUT_DATA
  FILE *file_in = fopen("input.bin", "wb");
  fwrite(input, 1, (size_t) nsamps * nchans * (in_nbits/8), file_in);
  fclose(file_in);
  #endif

  #if WRITE_OUTPUT_DATA
  FILE *file_out = fopen("output.bin", "wb");
  fwrite(output, 1, (size_t) nsamps_computed * dm_count * out_nbits/8, file_out);
  fclose(file_out);
  #endif

  tcheck->Pause();
  tbench->Pause();
  // Print timings
  printf("\n");
  printf("----------------------------- BENCHMARK TIMES  ----------------------------\n");
  std::cout << "Benchmark total time:      " << tbench->ToString() << " sec." << std::endl;
  std::cout << "Benchmark init time:       " << tinit->ToString() << " sec." << std::endl;
  std::cout << "Benchmark plan time:       " << tplan->ToString() << " sec." << std::endl;
  std::cout << "Benchmark execute time:    " << texecute->ToString() << " sec." << std::endl;
  std::cout << "Benchmark check time:      " << tcheck->ToString() << " sec." << std::endl;

  // Clean up
  free(output);
  free(input);
  printf("Benchmark finished\n");
  return 0;
}