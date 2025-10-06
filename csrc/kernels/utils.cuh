#pragma once

#include "exception.cuh"

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC) \
{ \
    constexpr int kLoopStride = kWarpSize * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * kWarpSize); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * kWarpSize, unrolled_values[__j]); \
    } \
    for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += kWarpSize) \
        ST_FUNC(__dst + __i, LD_FUNC(__src + __i)); \
}

#define UNROLLED_WARP_COPY_EMULATED(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC) \
{ \
    constexpr int kLoopStride = kEmulatedWarpSize * (UNROLL_FACTOR); \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC); \
    auto __dst = (DST); \
    for (int __i = (LANE_ID); __i < ((N) / kLoopStride) * kLoopStride; __i += kLoopStride) { \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            unrolled_values[__j] = LD_FUNC(__src + __i + __j * kEmulatedWarpSize); \
        _Pragma("unroll") \
        for (int __j = 0; __j < (UNROLL_FACTOR); ++ __j) \
            ST_FUNC(__dst + __i + __j * kEmulatedWarpSize, unrolled_values[__j]); \
    } \
    for (int __i = ((N) / kLoopStride) * kLoopStride + (LANE_ID); __i < (N); __i += kEmulatedWarpSize) \
        ST_FUNC(__dst + __i, LD_FUNC(__src + __i)); \
}
// HELPER FUNCTIONS #####################################################################################
#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

template <typename T>
DEVICE_INLINE T shfl_xor(
    const T val,
    int laneMask,
    int width = kWarpSize,
    uint64_t shfl_sync_mask = kFullWarpMask) {
#if defined(USE_ROCM) 
  return __shfl_xor(val, laneMask, width);
#else
  return __shfl_xor_sync(shfl_sync_mask, val, laneMask, width);
#endif
}

DEVICE_INLINE int shfl_sync(
    const int val,
    int srcLane = 0,
    int width = kWarpSize,
    uint64_t shfl_sync_mask = kFullWarpMask) {  // Let compiler deduce type
#if defined(USE_ROCM)
  return __shfl(val, srcLane, width);
#else
  return __shfl_sync(shfl_sync_mask, val, srcLane, width);
#endif
}

#ifdef USE_ROCM
DEVICE_INLINE int __any_sync(uint64_t mask, int predicate) {
  uint64_t predicate_bit_pattern = __ballot(predicate);
  return (predicate_bit_pattern & mask) > 0;
}

DEVICE_INLINE int __all_sync(uint64_t mask, int predicate) {
    uint64_t predicate_bit_pattern = __ballot(predicate);
    return (~predicate_bit_pattern & mask) == 0;
}
#endif

DEVICE_INLINE void syncwarp() {
#ifdef USE_ROCM
__builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
__builtin_amdgcn_wave_barrier();
__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");

//NOTE: This method will be tested for performance
//   // Performance - replace a block level __syncthreads with per CU
//   // __threadfence_block. It is a fine replacement for __syncwarp on AMD GPUs,
//   // it is because a. memory fencing: __threadfence_block ops. at CU level,
//   // same as __syncwarp at SM b. threads re-converge: wavefront run in
//   // lockstep, no need __syncwarp re-converge
//   __threadfence_block();
#else
  __syncwarp();
#endif
}
// ######################################################################################################

namespace deep_ep {

template <int kBytes>
struct VecInt {};
template<> struct VecInt<1> { using vec_t = int8_t; };
template<> struct VecInt<2> { using vec_t = int16_t; };
template<> struct VecInt<4> { using vec_t = int; };
template<> struct VecInt<8> { using vec_t = int64_t; };
template<> struct VecInt<16> { using vec_t = int4; };


// __device__ __forceinline__ void trap() {
//     asm("trap;");
// }
__device__ __forceinline__ void trap() {

#ifdef USE_ROCM
    abort();
#else
    asm("trap;");
#endif
}
// __device__ __forceinline__ void memory_fence() {
//    asm volatile("fence.acq_rel.sys;":: : "memory");
// }
__device__ __forceinline__ void memory_fence() {
#ifdef USE_ROCM
    __threadfence_system();
#else
    asm volatile("fence.acq_rel.sys;":: : "memory");
#endif
}

// __device__ __forceinline__ void memory_fence_gpu() {
//     asm volatile("fence.acq_rel.gpu;":: : "memory");
// }
__device__ __forceinline__ void memory_fence_gpu() {
#ifdef USE_ROCM
    __threadfence();
#else
    asm volatile("fence.acq_rel.gpu;":: : "memory");
#endif
}

// __device__ __forceinline__ void memory_fence_cta() {
//     
// }
__device__ __forceinline__ void memory_fence_cta() {
#ifdef USE_ROCM
    __threadfence_block();
#else
    asm volatile("fence.acq_rel.cta;":: : "memory");
#endif
}

// __device__  __forceinline__ void st_relaxed_sys_global(const int *ptr, int val) {
//    
// }
__device__  __forceinline__ void st_relaxed_sys_global(int *ptr, int val) {
#ifdef USE_ROCM
    __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("st.relaxed.sys.global.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
#endif
}

// __device__  __forceinline__ void st_release_sys_global(const int *ptr, int val) {
//     asm volatile("st.release.sys.global.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
// }
__device__  __forceinline__ void st_release_sys_global(const int *ptr, int val) {
#ifdef USE_ROCM
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("st.release.sys.global.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
#endif
}

// __device__  __forceinline__ void st_release_cta(const int *ptr, int val) {
//  asm volatile("st.release.cta.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
// }
__device__  __forceinline__ void st_release_cta(const int *ptr, int val) {
#ifdef USE_ROCM
    __hip_atomic_store(const_cast<int*>(ptr), val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
#else
    asm volatile("st.release.cta.s32 [%0], %1;"::"l"(ptr), "r"(val) : "memory");
#endif
}    

#ifdef USE_ROCM
__device__ __forceinline__ int ld_relaxed_sys_global(const int *ptr) {
    int ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}
__device__ __forceinline__ int ld_relaxed_sys_global(const uint64_t *ptr) {
    uint64_t ret;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    return ret;
}
#endif // USE_ROCM

// __device__ __forceinline__ int ld_acquire_sys_global(const int *ptr) {
//     int ret;
//     asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
//     return ret;
// }
__device__ __forceinline__ int ld_acquire_sys_global(const int *ptr) {
    int ret;
#ifdef USE_ROCM
    ret = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t *ptr) {
    uint64_t ret;
#ifdef USE_ROCM
    ret = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
    return ret;
}
//inter
__device__ __forceinline__ int ld_acquire_global(const int *ptr) {
    int ret;
#ifdef USE_ROCM
    ret = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#else    
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}
//not used
__device__ __forceinline__ int atomic_add_release_sys_global(const int* ptr, int value) {
    int ret;
#ifndef USE_ROCM
    asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
#endif
    return ret;
}
//inter
__device__ __forceinline__ int atomic_add_release_global(const int* ptr, int value) {
    int ret;
#ifdef USE_ROCM
    ret = __hip_atomic_fetch_add(const_cast<int*> (ptr), value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
#endif
return ret;
}
//inter
__device__ __forceinline__ int ld_acquire_cta(const int *ptr) {
    int ret;
#ifdef USE_ROCM
    ret = __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP);
#else
    asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}
//////////
//not used
__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t *ptr) {
    uint16_t ret;
#ifndef USE_ROCM
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
#endif
    return static_cast<uint8_t>(ret);
}
__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t *ptr) {
    uint16_t ret;
#ifndef USE_ROCM
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t *ptr) {
    uint32_t ret;
#ifndef USE_ROCM
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t *ptr) {
    uint64_t ret;
#ifndef USE_ROCM
    asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
    return ret;
}
////////////
// __device__  __forceinline__ int ld_volatile_global(const int *ptr) {
//     int ret;
//     asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
//     return ret;
// }
__device__  __forceinline__ int ld_volatile_global(const volatile int *ptr) {
    int ret;
#ifdef USE_ROCM
    //ret = *ptr;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}

// __device__  __forceinline__ float ld_volatile_global(const float *ptr) {
//     float ret;
//     asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
//     return ret;
// }
__device__  __forceinline__ float ld_volatile_global(const volatile float *ptr) {
    float ret;
#ifdef USE_ROCM
    //ret = *ptr;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
#endif 
    return ret;
}

// __device__  __forceinline__ int64_t ld_volatile_global(const int64_t *ptr) {
//     int64_t ret;
//     asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
//     return ret;
// }
__device__  __forceinline__ int64_t ld_volatile_global(const volatile int64_t *ptr) {
    int64_t ret;
#ifdef USE_ROCM
    //ret = *ptr;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
    return ret;
}
// __device__  __forceinline__ int64_t ld_volatile_global(const uint64_t *ptr) {
//     int64_t ret;
//     asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
//     return ret;
// }
__device__  __forceinline__ int64_t ld_volatile_global(const volatile uint64_t *ptr) {
    int64_t ret;
#ifdef USE_ROCM
    //ret = *ptr;
    ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
    return ret;
}

//TODO:: apply //"ld.global.nc.L1::no_allocate.L2::256B" on ROCM
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

// `ld.global.nc.L1::no_allocate` will be translated into `LDG.E.NA.[width].CONSTANT` in SASS
template <typename dtype_t>
__device__  __forceinline__ dtype_t ld_nc_global(const dtype_t *ptr) {
    auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr));
    return *reinterpret_cast<dtype_t*>(&ret);
}

// template <>
// __device__  __forceinline__ uint8_t ld_nc_global(const uint8_t *ptr) {
//     uint16_t ret;
//     // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit constraint letter (`h` below means unsigned 16-bit)
//     asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
//     return static_cast<uint8_t>(ret);
// }
template <>
__device__  __forceinline__ uint8_t ld_nc_global(const uint8_t *ptr) {
    uint8_t ret = *ptr;
#ifdef USE_ROCM 
    //ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    ret = __builtin_nontemporal_load(ptr);

#else
    asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
#endif
    return ret;
}

// template <>
// __device__  __forceinline__ int ld_nc_global(const int *ptr) {
//     int ret;
//     asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
//     return ret;
// }
template <>
__device__  __forceinline__ int ld_nc_global(const int *ptr) {
#ifdef USE_ROCM 
    int ret;
    //ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    ret = __builtin_nontemporal_load(ptr);
#else
    asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
#endif
    return ret;
}

// template <>
// __device__  __forceinline__ int64_t ld_nc_global(const int64_t *ptr) {
//     int64_t ret;
//     asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
//     return ret;
// }
template <>
__device__  __forceinline__ int64_t ld_nc_global(const int64_t *ptr) {
    int64_t ret;
#ifdef USE_ROCM
    // ret = *ptr;
    //ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    ret = __builtin_nontemporal_load(ptr);
#else
    asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
#endif
    return ret;
}

// template <>
// __device__  __forceinline__ float ld_nc_global(const float *ptr) {
//     float ret;
//     asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
//     return ret;
// }
template <>
__device__  __forceinline__ float ld_nc_global(const float *ptr) {
    float ret;
#ifdef USE_ROCM
    // ret = *ptr;
    //ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    ret = __builtin_nontemporal_load(ptr);
#else
    asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
#endif
    return ret;
}

// template <>
// __device__  __forceinline__ int2 ld_nc_global(const int2 *ptr) {
//     int2 ret;
//     asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
//     return ret;
// }
template <>
__device__  __forceinline__ int2 ld_nc_global(const int2 *ptr) {
    int2 ret;
#ifdef USE_ROCM
    // ret = *ptr;
    //ret = __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    int x,y;
    x = __builtin_nontemporal_load(&(ptr->x));
    y = __builtin_nontemporal_load(&(ptr->y));
    ret = {x,y};
#else
    asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
#endif
    return ret;
}

// template <>
// __device__  __forceinline__ int4 ld_nc_global(const int4 *ptr) {
//     int4 ret;
//     asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
//             : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
//     return ret;
// }
//TODO:: volatile missed
template <>
__device__  __forceinline__ int4 ld_nc_global(const int4 *ptr) {
     int4 ret;
#ifdef USE_ROCM
    //TODO:: causes slow down
    int x,y,z,w;
    x = __builtin_nontemporal_load(&(ptr->x));
    y = __builtin_nontemporal_load(&(ptr->y));
    z = __builtin_nontemporal_load(&(ptr->z));
    w = __builtin_nontemporal_load(&(ptr->w));
    ret = {x,y,z,w};
    //ret = *ptr;
#else
    asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w) : "l"(ptr));
#endif
    return ret;
}

////////////////// used in ibgda
__device__ __forceinline__ void st_na_relaxed(const uint8_t *ptr, uint8_t val) {
#ifdef USE_ROCM
    uint8_t* non_const_ptr = const_cast<uint8_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(static_cast<uint16_t>(val)));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t *ptr, uint16_t val) {
#ifdef USE_ROCM
    uint16_t* non_const_ptr = const_cast<uint16_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
     asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t *ptr, uint32_t val) {
#ifdef USE_ROCM
    uint32_t* non_const_ptr = const_cast<uint32_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
     asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const int *ptr, int val) {
#ifdef USE_ROCM
    int* non_const_ptr = const_cast<int*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
     asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_relaxed(const int4 *ptr, int4 val) {
#ifdef USE_ROCM
    int4* non_const_ptr = const_cast<int4*>(ptr);
    non_const_ptr->x = val.x;
    non_const_ptr->y = val.y;
    non_const_ptr->z = val.z;
    non_const_ptr->w = val.w;
#else
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
            : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

__device__ __forceinline__ void st_na_release(const int *ptr, int val) {
#ifdef USE_ROCM
    int* non_const_ptr = const_cast<int*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_release(const uint32_t *ptr, uint32_t val) {
#ifdef USE_ROCM
    uint32_t* non_const_ptr = const_cast<uint32_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
#endif
}

__device__ __forceinline__ void st_na_release(const uint64_t *ptr, uint64_t val) {
#ifdef USE_ROCM
    uint64_t* non_const_ptr = const_cast<uint64_t*>(ptr);
    __hip_atomic_store(non_const_ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
#endif
}
/////////////////////////////////
// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate" 
#else
#define ST_NA_FUNC "st.global"
#endif

//TODO:: apply "st.global.L1::no_allocate" in ROCM
template <typename dtype_t>
__device__  __forceinline__ void st_na_global(const dtype_t *ptr, const dtype_t& value) {
    st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(ptr),
                 *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t*>(&value));
}

template <>
__device__  __forceinline__ void st_na_global(const int *ptr, const int& value) {
#ifdef USE_ROCM
    int* non_const_ptr = const_cast<int*>(ptr);
    *non_const_ptr = value;
#else
    asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
#endif
}

template <>
__device__  __forceinline__ void st_na_global(const int64_t *ptr, const int64_t& value) {
#ifdef USE_ROCM
    int64_t* non_const_ptr = const_cast<int64_t*>(ptr);
    *non_const_ptr = value;
#else
    asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
#endif
}

template <>
__device__  __forceinline__ void st_na_global(const float *ptr, const float& value) {
#ifdef USE_ROCM
    float* non_const_ptr = const_cast<float*>(ptr);
    *non_const_ptr = value;
#else
    asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
#endif
}

template <>
__device__  __forceinline__ void st_na_global(const int4 *ptr, const int4& value) {
#ifdef USE_ROCM
    int4* non_const_ptr = const_cast<int4*>(ptr);
    *non_const_ptr = value;
#else
    asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};"
            ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w));
#endif
}

template <typename dtype_t>
__host__ __device__ dtype_t cell_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ dtype_t align(dtype_t a, dtype_t b) {
    return cell_div<dtype_t>(a, b) * b;
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id,
                                                       int& token_start_idx, int& token_end_idx) {
    int num_tokens_per_sm = cell_div(num_tokens, num_sms);
    token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
    token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    dtype_b_t packed;
    auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
    unpacked_ptr[0] = x, unpacked_ptr[1] = y;
    return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
    EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
    auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
    x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename dtype_t>
__device__ __forceinline__ dtype_t broadcast(dtype_t& ptr, int src_lane_idx) {
    EP_STATIC_ASSERT(sizeof(dtype_t) % sizeof(int) == 0, "");
    auto send_int_values = reinterpret_cast<int*>(&ptr);
    int recv_int_values[sizeof(dtype_t) / sizeof(int)];
    #pragma unroll
    for (int i = 0; i < sizeof(dtype_t) / sizeof(int); ++ i)
        recv_int_values[i] = shfl_sync(send_int_values[i], src_lane_idx);
    return *reinterpret_cast<dtype_t*>(recv_int_values);
}

__forceinline__ __device__ int warp_reduce_sum(int value) {
    if constexpr (kWarpSize==64) 
        value += shfl_xor<int>(value, 32);
    value += shfl_xor<int>( value, 16);
    value += shfl_xor<int>( value, 8);
    value += shfl_xor<int>(value, 4);
    value += shfl_xor<int>( value, 2);
    value += shfl_xor<int>(value, 1);
    return value;
}

__forceinline__ __device__ float half_warp_reduce_max(float value) {
    if constexpr (kWarpSize==64) 
        value = max(value, shfl_xor<float>(value, 16));
    value = max(value, shfl_xor<float>(value, 8));
    value = max(value, shfl_xor<float>(value, 4));
    value = max(value, shfl_xor<float>( value, 2));
    value = max(value, shfl_xor<float>(value, 1));
    return value;
}

#ifdef USE_ROCM
__forceinline__ __device__ float quarter_warp_reduce_max(float value) {
    value = max(value, shfl_xor<float>(value, 8));
    value = max(value, shfl_xor<float>(value, 4));
    value = max(value, shfl_xor<float>( value, 2));
    value = max(value, shfl_xor<float>(value, 1));
    return value;
}
#endif

__forceinline__ __device__ int get_lane_id() {
    int lane_id;
#ifdef USE_ROCM
    lane_id = threadIdx.x % kWarpSize;
#else
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
#endif
    return lane_id;
}

template <int kNumRanks>
__forceinline__ __device__ void move_fifo_slots(int &head) {
    head = (head + kNumRanks) % NUM_MAX_FIFO_SLOTS;
}

template <int kNumRanks>
__device__ __forceinline__ bool not_finished(int *task, int expected) {
    auto result = false;
    auto lane_id = threadIdx.x % kWarpSize;
    if (lane_id < kNumRanks)
        result = ld_volatile_global(task + lane_id) != expected;
    return __any_sync(kFullWarpMask, result);
}

template <int kNumRanks>
__forceinline__ __device__ void
timeout_check(int **task_fifo_ptrs, int head, int rank, int expected, int tag = 0) {
    auto start_time = wall_clock64();
    while (not_finished<kNumRanks>(task_fifo_ptrs[rank] + head, expected)) {
        long long int elapsed_time = wall_clock64() > start_time ? wall_clock64() - start_time : 0;
        if (elapsed_time > NUM_TIMEOUT_CYCLES and threadIdx.x == 0) {
            printf("DeepEP timeout check failed: %d (rank = %d)\n", tag, rank);
            trap();
        }
    }
}

template <int kNumRanks>
__forceinline__ __device__ void
barrier_device(int **task_fifo_ptrs, int head, int rank, int tag = 0) {
    auto thread_id = static_cast<int>(threadIdx.x);
    EP_DEVICE_ASSERT(kNumRanks <= kWarpSize);

    if (thread_id < kNumRanks) {
        atomicAdd_system(task_fifo_ptrs[rank] + head + thread_id, FINISHED_SUM_TAG);
        memory_fence();
        atomicSub_system(task_fifo_ptrs[thread_id] + head + rank, FINISHED_SUM_TAG);
    }
    timeout_check<kNumRanks>(task_fifo_ptrs, head, rank, 0, tag);
}

} // namespace deep_ep
