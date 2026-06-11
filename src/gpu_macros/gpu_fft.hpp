#ifndef GPUFFT_H
#define GPUFFT_H

#ifdef USE_CUDA
  // NVIDIA / CUFFT :

  #include <cufftw.h>
  #include <cufft.h>
  #define gpufftResult   cufftResult
  #define gpufftDestroy  cufftDestroy
  #define GPUFFT_SUCCESS CUFFT_SUCCESS
  #define gpufftComplex  cufftComplex
  #define gpufftPlanMany cufftPlanMany
  #define gpufftHandle   cufftHandle
  #define gpufftPlan2d   cufftPlan2d
  #define gpufftExecC2C  cufftExecC2C
  #define GPUFFT_C2C     CUFFT_C2C
  #define GPUFFT_FORWARD CUFFT_FORWARD
  #define gpufftPlan1d  cufftPlan1d
  #define GPUFFT_R2C CUFFT_R2C
  #define GPUFFT_C2R CUFFT_C2R
  #define gpufftSetStream(plan, stream) GPUFFT_CHECK(cufftSetStream((plan), (stream)))
  #define gpufftReal cufftReal
  #define gpufftExecR2C(...) GPUFFT_CHECK(cufftExecR2C(__VA_ARGS__))
  #define gpufftExecC2R(...) GPUFFT_CHECK(cufftExecC2R(__VA_ARGS__))
#endif

#ifdef USE_HIP
  // AMD / HIP :
  #include <hipfft/hipfft.h>
  
  #define gpufftResult   hipfftResult
  #define gpufftDestroy  hipfftDestroy
  #define GPUFFT_SUCCESS HIPFFT_SUCCESS
  #define gpufftComplex  hipfftComplex
  #define gpufftPlanMany hipfftPlanMany
  #define gpufftHandle   hipfftHandle
  #define gpufftPlan2d   hipfftPlan2d
  #define gpufftExecC2C  hipfftExecC2C
  #define GPUFFT_C2C     HIPFFT_C2C
  #define GPUFFT_FORWARD HIPFFT_FORWARD
  #define gpufftPlan1d  hipfftPlan1d
  #define GPUFFT_R2C HIPFFT_R2C
  #define GPUFFT_C2R HIPFFT_C2R
  #define gpufftSetStream(plan, stream) GPUFFT_CHECK(hipfftSetStream((plan), (stream)))
  #define gpufftReal hipfftReal
  #define gpufftExecR2C(...) GPUFFT_CHECK(hipfftExecR2C(__VA_ARGS__))
  #define gpufftExecC2R(...) GPUFFT_CHECK(hipfftExecC2R(__VA_ARGS__))
#endif

#define GPUFFT_CHECK(x) do { \
    gpufftResult r = (x); \
    if (r != GPUFFT_SUCCESS) { \
        fprintf(stderr, "GPU FFT error: %d (%s:%d)\n", r, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#ifdef USE_OPENMP
  #include <fftw3.h>
  typedef float        gpufftReal;
  typedef fftwf_complex gpufftComplex;
  #define GPUFFT_R2C 0
  #define GPUFFT_C2R 1
  #define GPUFFT_SUCCESS 0
  typedef int gpufftResult;

  struct gpufftHandle_t {
      int type;        // GPUFFT_R2C or GPUFFT_C2R
      int n;           // logical transform length
      int batch;
      int istride, idist, ostride, odist;
  };
  typedef gpufftHandle_t gpufftHandle;

  static inline gpufftResult gpufftPlanMany(
      gpufftHandle* plan, int rank, int* n,
      int* inembed, int istride, int idist,
      int* onembed, int ostride, int odist,
      int type, int batch) {
      (void)rank; (void)inembed; (void)onembed;
      *plan = gpufftHandle{ type, n[0], batch, istride, idist, ostride, odist };
      return GPUFFT_SUCCESS;
  }
  static inline gpufftResult gpufftPlan1d(gpufftHandle* plan, int n, int type, int batch) {
      *plan = gpufftHandle{ type, n, batch, 1, n, 1, (type==GPUFFT_R2C ? n/2+1 : n) };
      return GPUFFT_SUCCESS;
  }
  #define gpufftSetStream(plan, stream) GPUFFT_SUCCESS

  static inline gpufftResult gpufftExecR2C(gpufftHandle p, gpufftReal* in, gpufftComplex* out) {
      int n = p.n;
      fftwf_plan fp = fftwf_plan_many_dft_r2c(
          1, &n, p.batch,
          in,  nullptr, p.istride, p.idist,
          out, nullptr, p.ostride, p.odist,
          FFTW_ESTIMATE);
      fftwf_execute(fp);
      fftwf_destroy_plan(fp);
      return GPUFFT_SUCCESS;
  }
  static inline gpufftResult gpufftExecC2R(gpufftHandle p, gpufftComplex* in, gpufftReal* out) {
      int n = p.n;
      fftwf_plan fp = fftwf_plan_many_dft_c2r(
          1, &n, p.batch,
          in,  nullptr, p.istride, p.idist,
          out, nullptr, p.ostride, p.odist,
          FFTW_ESTIMATE);
      fftwf_execute(fp);
      fftwf_destroy_plan(fp);
      return GPUFFT_SUCCESS;
  }
  #define gpufftDestroy(plan) GPUFFT_SUCCESS
#endif

#endif