#ifndef GPU_VECTYPES_HPP
#define GPU_VECTYPES_HPP
// Host stand-ins for CUDA vector types, used by the FDD kernels under USE_OPENMP.
#ifdef USE_OPENMP
struct float2 { float x, y; };
static inline float2 make_float2(float x, float y) { return float2{x, y}; }
#endif
#endif // GPU_VECTYPES_HPP
