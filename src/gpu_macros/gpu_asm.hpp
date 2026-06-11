#ifndef GPU_ASM_HPP
#define GPU_ASM_HPP

#ifdef USE_CUDA
    #define gpu_fmul_rn(out, a, b) \
    asm volatile ("mul.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b))

    #define gpu_fma_rn_ftz(out, a, b, c) \
    asm volatile ("fma.rn.ftz.f32 %0, %1, %2, %3;" \
         : "=f"(out) \
         : "f"(a), "f"(b), "f"(c))

    #define gpu_sin(r,a) \
    asm volatile ("sin.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a))

    #define gpu_cos(r,a) \
    asm volatile ("cos.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a))
#endif

#ifdef USE_HIP
    #define gpu_fmul_rn(out, a, b) \
    ((out) = __fmul_rn((a), (b)))

    #define gpu_fma_rn_ftz(out, a, b, c) \
    ((out) = __fmaf_rn((a), (b), (c)))

    #define gpu_sin(r,a) \
    ((r) = __sinf((a)))

    #define gpu_cos(r,a) \
    ((r) = __cosf((a)))
#endif

#ifdef USE_OPENMP
  #include <cmath>
  #define gpu_fmul_rn(out, a, b)        ((out) = (a) * (b))
  #define gpu_fma_rn_ftz(out, a, b, c)  ((out) = std::fma((a), (b), (c)))
  #define gpu_sin(r, a)                 ((r) = std::sin((a)))
  #define gpu_cos(r, a)                 ((r) = std::cos((a)))
#endif

#endif