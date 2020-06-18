#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#include <DedispPlan.hpp>

struct header {
  int64_t headersize,buffersize;
  int nchan,nsamp,nbit,nif;
  int machine_id,telescope_id,nbeam,ibeam,sumif;
  double tstart,tsamp,fch1,foff;
  double src_raj,src_dej,az_start,za_start;
  char source_name[80],ifstream[8],inpfile[80];
};
struct header read_header(FILE *file);
static char *backend_name(int machine_id);
static char *telescope_name(int telescope_id);
void writeinf(struct header h, char *outstem, float dm,int numout);

// Decimate the timeseries in time. Reuses input array                          
void decimate_timeseries(dedisp_byte *z,int nx,int ny,int mx)
{
  int64_t i,j,k,l;
  float ztmp;
  dedisp_byte c;

  for (j=0,l=0;j<ny;j+=mx,l++) {
    for (i=0;i<nx;i++) {
      ztmp=0.0;
      for (k=0;k<mx;k++)
        ztmp+=(float) z[i+nx*(j+k)];
      ztmp/=mx;
      if (ztmp>255.0)
	c=(dedisp_byte) 255;
      else if (ztmp<0.0)
	c=(dedisp_byte) 0;
      else
	c=(dedisp_byte) ztmp;
      z[i+nx*l]=c;
    }
  }

  return;
}

void usage(void)
{
  printf("Usage: dedisp_fil -f <file> -D [GPU device] -r [DM start,end] -s [DM step]\n\n");
  printf("-f <FILE file>      SIGPROC filterbank file to read as input\n");
  printf("-o <output prefix>  Output prefix [test]\n");
  printf("-D [GPU device]     GPU device number to use (default: 0)\n");
  printf("-r [DM start(,end)] Start (and end) values of DM range to dedisperse (default: 0,50)\n");
  printf("-s [DM step]        Linear DM step to use. If not provided, optimal DM trials are computed\n");
  printf("-n [ntrails]        Number of DM trails.\n");
  printf("-d [decimate]       Decimate timeseries by this factor (default: 1)\n");
  printf("-N [samples]        Number of samples to write (default: all)\n");
  printf("-w [pulse width]    Expected intrinsic pulse width for optimal DM trials [default: 4.0us]\n");
  printf("-t [tolerance]      Smearing tolerance factor between DM trials [default: 1.25]\n");
  printf("-g [gulp size]      Gulp size [default: 65536]\n");
  printf("-q                  Quiet; no information to screen\n");
  printf("-h                  This help\n");

  return;
}

int main(int argc,char *argv[])
{
  int i,device_id=0,verbose=1;
  struct header h;
  FILE *file;
  dedisp_byte *input=0;
  dedisp_float *output=0;
  dedisp_float *dmlist;
  dedisp_size dm_count=0,max_delay,nsamp_computed,ndec=1;
  dedisp_error error;
  dedisp_float dm_start=0.0,dm_end=50.0,dm_step=0.0,pulse_width=4.0,dm_tol=1.25;
  dedisp_size nbits=32,gulp_size=65536;
  clock_t startclock;
  int arg=0;
  char *filename=NULL,prefix[128]="test";
  int numout=0;

  // Decode options
  if (argc>1) {
    while ((arg=getopt(argc,argv,"f:D:hr:s:w:t:qo:d:n:N:g:"))!=-1) {
      switch(arg) {
	
      case 'g':
	gulp_size=(dedisp_size) atoi(optarg);
	break;

      case 'f':
	filename=optarg;
	break;
	
      case 'o':
	strcpy(prefix,optarg);
	break;
	
      case 'r':
	if (strchr(optarg,',')!=NULL)
	  sscanf(optarg,"%f,%f",&dm_start,&dm_end);
	else
	  dm_start=atof(optarg);
	break;
	
      case 's':
	dm_step=atof(optarg);
	break;
	
      case 'd':
	ndec=(dedisp_size) atoi(optarg);
	break;
	
      case 'n':
	dm_count=(dedisp_size) atoi(optarg);
	break;
	
      case 'N':
	numout=atoi(optarg);
	break;
	
      case 'D':
	device_id=atoi(optarg);
	break;
	
      case 'q':
	verbose=0;
	break;
	
      case 'h':
	usage();
	return 0;
	break;
	
      default:
	usage();
	return 0;
      }
    }
  } else {
    usage();
    return 0;
  }

  // Set end DM
  if (dm_count!=0 && dm_step>0.0) {
    dm_end=dm_start+dm_count*dm_step;
  } else {
    fprintf(stderr,"Error parsing DM range. Provide start,end,step or start,step,numdms.\n",filename);
    return -1;
  }

  // Open file 
  file=fopen(filename,"r");
  if (file==NULL) {
    fprintf(stderr,"Error reading file %s\n",filename);
    return -1;
  }

  // Read header
  h=read_header(file);

  // Print information
  if (verbose) {
    printf("----------------------------- INPUT DATA ---------------------------------\n");
    printf("Frequency of highest channel              : %f MHz\n",h.fch1);
    printf("Bandwidth                                 : %f MHz\n",fabs(h.foff)*h.nchan);
    printf("Number of channels (channel width)        : %d (%f MHz)\n",h.nchan,fabs(h.foff));
    printf("Sample time                               : %f us\n",h.tsamp*1e6*ndec);
    printf("Observation duration                      : %f s (%d samples)\n",h.tsamp*h.nsamp,h.nsamp/ndec);
    printf("Number of polarizations/bit depth         : %d/%d\n",h.nif,h.nbit);
    printf("Input data array size                     : %lu MB\n",h.buffersize/(1<<20));
    printf("Header size                               : %lu bytes\n",h.headersize);
    printf("\n");
  }

  // Exit on wrong type of input data
  if (h.nif!=1) {
    fprintf(stderr,"Wrong number of polarizations (not 1). Exiting.\n");
    return -1;
  }
  if (h.nbit!=8) {
    fprintf(stderr,"Wrong bit depth (not 8). Exiting.\n");
    return -1;
  }

  // Read buffer
  if (verbose) printf("Reading input file\n");
  startclock=clock();
  input=(dedisp_byte *) malloc(sizeof(dedisp_byte)*h.buffersize);
  fread(input,sizeof(dedisp_byte),h.buffersize,file);
  if (verbose) printf("Reading input file took %.2f seconds\n",(double)(clock()-startclock)/CLOCKS_PER_SEC);

  // Decimate if required
  if (ndec>1) {
    if (verbose) printf("Decimate timeseries by factor %d\n",(int) ndec);
    startclock=clock();
    decimate_timeseries(input,h.nchan,h.nsamp,ndec);
    if (verbose) printf("Decimating timeseries took %.2f seconds\n",(double)(clock()-startclock)/CLOCKS_PER_SEC);

    // Adjust values
    h.tsamp*=ndec;
    h.nsamp/=ndec;
  }

  // Close file;
  fclose(file);

  // Create a dedispersion plan
  if (verbose) printf("Creating dedispersion plan\n");
  DedispPlan plan(h.nchan,h.tsamp,h.fch1,h.foff);

  // Intialize GPU
  if (verbose) printf("Intializing GPU (device %d)\n",device_id);
  plan.set_device(device_id);

  // Generate a list of dispersion measures for the plan
  if (dm_step==0) {
    if (verbose) printf("Generating optimal DM trials\n");
    plan.generate_dm_list(dm_start,dm_end,pulse_width,dm_tol);
  } else {
  // Generate a list of dispersion measures for the plan
    if (verbose) printf("Generating linear DM trials\n");
    if (dm_count==0)
      dm_count=(int) ceil((dm_end-dm_start)/dm_step);
    dmlist=(dedisp_float *) calloc(sizeof(dedisp_float),dm_count);
    for (i=0;i<dm_count;i++)
      dmlist[i]=(dedisp_float) dm_start+dm_step*i;
    plan.set_dm_list(dmlist,dm_count);
  }

  // Get specifics of the computed dedispersion plan
  dm_count=plan.get_dm_count();
  max_delay=plan.get_max_delay();
  nsamp_computed=h.nsamp-max_delay;
  dmlist=(float *)plan.get_dm_list();

  // Print information
  if (verbose) {
    printf("----------------------------- DM COMPUTATIONS  ----------------------------\n");
    printf("Computing %ld DMs from %f to %f pc/cm^3\n",dm_count,dmlist[0],dmlist[dm_count-1]);
    printf("Max DM delay is %ld samples (%f seconds)\n",max_delay,max_delay*h.tsamp);
    printf("Computing %ld out of %d total samples (%.2f%% efficiency)\n",nsamp_computed,h.nsamp,100.0*(dedisp_float)nsamp_computed/h.nsamp);
    if (dm_step==0.0)
      printf("Pulse width: %f, DM tolerance: %f\n",pulse_width,dm_tol);
    printf("Output data array size : %ld MB\n",(dm_count*nsamp_computed*(nbits/8))/(1<<20));
    printf("\n");
  }

  // Allocate space for the output data
  output=(dedisp_float *) malloc(nsamp_computed*dm_count*sizeof(dedisp_float));
  if (output==NULL) {
    printf("\nERROR: Failed to allocate output array\n");
    return -1;
  }

  // Setting maximum gulp_size
  plan.set_gulp_size(gulp_size);
  printf("Current gulp_size = %d\n", (int)plan.get_gulp_size());

  // Perform computation
  if (verbose) printf("Dedispersing on the GPU\n");
  startclock=clock();
  plan.execute(h.nsamp,input,h.nbit,(dedisp_byte *)output,nbits,DEDISP_USE_DEFAULT);
  error=dedisp_sync();
  if (error!=DEDISP_NO_ERROR) {
    printf("\nERROR: Failed to synchronize: %s\n",dedisp_get_error_string(error));
    return -1;
  }
  if (verbose) printf("Dedispersion took %.2f seconds\n",(double)(clock()-startclock)/CLOCKS_PER_SEC);

  // Output length
  if (numout==0 || numout>nsamp_computed)
    numout=nsamp_computed;
  
  // Ensure numout divisible by 2
  if (numout%2!=0)
    numout--;

  // Write output DM trials
  startclock=clock();
  for (i=0;i<dm_count;i++) {
    // Generate output file name
    sprintf(filename,"%s_DM%.3f.dat",prefix,dmlist[i]);
    file=fopen(filename,"w");
    if (file==NULL) {
      fprintf(stderr,"Error opening %s\n",filename);
      return -1;
    }

    // Write buffer
    fwrite(output+i*nsamp_computed,sizeof(dedisp_float),numout,file);

    // Close file
    fclose(file);

    // Write inf file
    writeinf(h,prefix,dmlist[i],numout);
  }
  if (verbose) printf("Writing DM trials took %.2f seconds\n",(double)(clock()-startclock)/CLOCKS_PER_SEC);

  // Clean up
  free(input);
  free(output);

  return 0;
}

// Read SIGPROC filterbank header
struct header read_header(FILE *file)
{
  int nchar,nbytes=0;
  char string[80];
  struct header h;

  // Read header parameters
  for (;;) {
    // Read string size
    strcpy(string,"ERROR");
    fread(&nchar,sizeof(int),1,file);
    
    // Skip wrong strings
    if (!(nchar>1 && nchar<80)) 
      continue;

    // Increate byte counter
    nbytes+=nchar;

    // Read string
    fread(string,nchar,1,file);
    string[nchar]='\0';
    
    // Exit at end of header
    if (strcmp(string,"HEADER_END")==0)
      break;
    
    // Read parameters
    if (strcmp(string, "tsamp")==0) 
      fread(&h.tsamp,sizeof(double),1,file);
    else if (strcmp(string,"tstart")==0) 
      fread(&h.tstart,sizeof(double),1,file);
    else if (strcmp(string,"fch1")==0) 
      fread(&h.fch1,sizeof(double),1,file);
    else if (strcmp(string,"foff")==0) 
      fread(&h.foff,sizeof(double),1,file);
    else if (strcmp(string,"nchans")==0) 
      fread(&h.nchan,sizeof(int),1,file);
    else if (strcmp(string,"nifs")==0) 
      fread(&h.nif,sizeof(int),1,file);
    else if (strcmp(string,"nbits")==0) 
      fread(&h.nbit,sizeof(int),1,file);
    else if (strcmp(string,"nsamples")==0) 
      fread(&h.nsamp,sizeof(int),1,file);
    else if (strcmp(string,"az_start")==0) 
      fread(&h.az_start,sizeof(double),1,file);
    else if (strcmp(string,"za_start")==0) 
      fread(&h.za_start,sizeof(double),1,file);
    else if (strcmp(string,"src_raj")==0) 
      fread(&h.src_raj,sizeof(double),1,file);
    else if (strcmp(string,"src_dej")==0) 
      fread(&h.src_dej,sizeof(double),1,file);
    else if (strcmp(string,"telescope_id")==0) 
      fread(&h.telescope_id,sizeof(int),1,file);
    else if (strcmp(string,"machine_id")==0) 
      fread(&h.machine_id,sizeof(int),1,file);
    else if (strcmp(string,"nbeams")==0) 
      fread(&h.nbeam,sizeof(int),1,file);
    else if (strcmp(string,"ibeam")==0) 
      fread(&h.ibeam,sizeof(int),1,file);
    else if (strcmp(string,"source_name")==0) 
      strcpy(h.source_name,string);
  }

  // Get header and buffer sizes
  h.headersize=(int64_t) ftell(file);
  fseek(file,0,SEEK_END);
  h.buffersize=ftell(file)-h.headersize;
  h.nsamp=h.buffersize/(h.nchan*h.nif*h.nbit/8);

  // Reset file pointer to start of buffer
  rewind(file);
  fseek(file,h.headersize,SEEK_SET);

  return h;
}

static char *telescope_name(int telescope_id)
{
  char *telescope, string[80];
  switch (telescope_id) {
  case 0:
    strcpy(string, "Fake");
    break;
  case 1:
    strcpy(string, "Arecibo");
    break;
  case 2:
    strcpy(string, "Ooty");
    break;
  case 3:
    strcpy(string, "Nancay");
    break;
  case 4:
    strcpy(string, "Parkes");
    break;
  case 5:
    strcpy(string, "Jodrell");
    break;
  case 6:
    strcpy(string, "GBT");
    break;
  case 7:
    strcpy(string, "GMRT");
    break;
  case 8:
    strcpy(string, "Effelsberg");
    break;
  case 9:
    strcpy(string, "ATA");
    break;
  case 10:
    strcpy(string, "UTR-2");
    break;
  case 11:
    strcpy(string, "LOFAR");
    break;
  case 12:
    strcpy(string, "FR606");
    break;
  case 13:
    strcpy(string, "DE601");
    break;
  case 14:
    strcpy(string, "UK608");
    break;
  default:
    strcpy(string, "???????");
    break;
  }
  telescope = (char *) calloc(strlen(string) + 1, 1);
  strcpy(telescope, string);
  return telescope;
}

static char *backend_name(int machine_id)
{
  char *backend, string[80];
  switch (machine_id) {
  case 0:
    strcpy(string, "FAKE");
    break;
  case 1:
    strcpy(string, "PSPM");
    break;
  case 2:
    strcpy(string, "WAPP");
    break;
  case 3:
    strcpy(string, "AOFTM");
    break;
  case 4:
    strcpy(string, "BPP");
    break;
  case 5:
    strcpy(string, "OOTY");
    break;
  case 6:
    strcpy(string, "SCAMP");
    break;
  case 7:
    strcpy(string, "SPIGOT");
    break;
  case 10:
    strcpy(string, "ARTEMIS");
    break;
  case 11:
    strcpy(string, "Cobalt");
    break;
  default:
    strcpy(string, "????");
    break;
  }
  backend = (char *) calloc(strlen(string) + 1, 1);
  strcpy(backend, string);
  return backend;
}

// writes out .inf file
void writeinf(struct header h, char *outstem, float dm,int numout)
{
  char outname[1024];
  char tmp1[100], tmp2[100];
  int itmp;
  int ra_h, ra_m, dec_d, dec_m;
  double ra_s, dec_s;
  FILE *infofile;
  char sign;

  sprintf(outname, "%s_DM%.3f.inf", outstem, dm);

  // first check if file already exists                                                                                                                                                
  // if it does, then return                                                                                                                                                           
  // struct stat info;                                                                                                                                                                 
  // if (stat(outname, &info) == 0) return;                                                                                                                                            

  if ((infofile=fopen(outname, "w")) == NULL) {
    fprintf(stderr, "Error opening output inf-file!\n");
    exit(1);
  }

  fprintf(infofile, " Data file name without suffix          =  %s_DM%.3f\n", outstem, dm);
  fprintf(infofile, " Telescope used                         =  %s\n", telescope_name(h.telescope_id));
  fprintf(infofile, " Instrument used                        =  %s\n", backend_name(h.machine_id));
  fprintf(infofile, " Object being observed                  =  %s\n", h.source_name);
  ra_h = (int) floor(h.src_raj / 10000.0);
  ra_m = (int) floor((h.src_raj - ra_h * 10000) / 100.0);
  ra_s = h.src_raj - ra_h * 10000 - ra_m * 100;
  dec_d = (int) floor(fabs(h.src_dej) / 10000.0);
  dec_m = (int) floor((fabs(h.src_dej) - dec_d * 10000) / 100.0);
  dec_s = fabs(h.src_dej) - dec_d * 10000 - dec_m * 100;
  //  if (h.src_dej < 0.0) dec_d = -dec_d;
  sign=(h.src_dej<0.0 ? '-' : ' ');
  fprintf(infofile, " J2000 Right Ascension (hh:mm:ss.ssss)  =  %02d:%02d:%02f\n", ra_h, ra_m, ra_s);
  fprintf(infofile, " J2000 Declination     (dd:mm:ss.ssss)  = %c%02d:%02d:%s%f\n", sign,dec_d, dec_m, dec_s < 10 ? "0" : "", dec_s);
  fprintf(infofile, " Data observed by                       =  Unknown\n");
  sprintf(tmp1, "%.15f", h.tstart - (int) floor(h.tstart));
  sscanf(tmp1, "%d.%s", &itmp, tmp2);
  fprintf(infofile, " Epoch of observation (MJD)             =  %d.%s\n", (int) floor(h.tstart), tmp2);
  fprintf(infofile, " Barycentered?           (1=yes, 0=no)  =  0\n");
  fprintf(infofile, " Number of bins in the time series      =  %d\n",numout);
  fprintf(infofile, " Width of each time series bin (sec)    =  %.15g\n", h.tsamp);
  fprintf(infofile, " Any breaks in the data? (1=yes, 0=no)  =  0\n");
  fprintf(infofile, " Type of observation (EM band)          =  Radio\n");
  fprintf(infofile, " Beam diameter (arcsec)                 =  3600\n");
  fprintf(infofile, " Dispersion measure (cm-3 pc)           =  %.12g\n", dm);
  fprintf(infofile, " Central freq of low channel (Mhz)      =  %.12g\n", h.fch1 - (h.nchan - 1) * fabs(h.foff));
  fprintf(infofile, " Total bandwidth (Mhz)                  =  %.12g\n", fabs(h.foff) * h.nchan);
  fprintf(infofile, " Number of channels                     =  %d\n", h.nchan);
  fprintf(infofile, " Channel bandwidth (Mhz)                =  %.12g\n", fabs(h.foff));
  fprintf(infofile, " Data analyzed by                       =  Unknown\n");
  fprintf(infofile, " Any additional notes:\n    Input filterbank samples have %d bits.\n", h.nbit);

  fclose(infofile);
}
