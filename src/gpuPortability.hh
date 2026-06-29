#ifndef GPUPORTABILITY_HH
#define GPUPORTABILITY_HH

#if defined __CUDACC__ || defined TARGET_NVIDIA
    #define __DO_CUDA
    #define __PREFIX cuda
    #define HAVE_UVM
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
#elif defined __HIPCC__ || defined TARGET_AMD
    #define __DO_HIP
    #define __PREFIX hip
    #define HAVE_UVM
    #define __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
#else
    #define __PREFIX invalid
#endif

#if defined HAVE_CUDA || defined HAVE_HIP
    #define GPU_NATIVE
#endif


#ifdef __DO_CUDA
#endif

#ifdef __DO_HIP
#endif

#if defined HAVE_UVM
    #define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
#else
    #define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
#endif  

#define CONCAT_(A, B) A ## B
#define CONCAT(A1, B1) CONCAT_(A1, B1)

#define gpuMallocManaged      CONCAT(__PREFIX, MallocManaged)
#define gpuFree               CONCAT(__PREFIX, Free)
#define gpuDeviceSynchronize  CONCAT(__PREFIX, DeviceSynchronize)
#define gpuGetDeviceCount     CONCAT(__PREFIX, GetDeviceCount)
#define gpuSetDevice          CONCAT(__PREFIX, SetDevice)
#define gpuPeekAtLastError    CONCAT(__PREFIX, PeekAtLastError)

// --- Explicit-device-memory and stream API (added for the GPU-resident /
// --- RCCL communication port).  These map to hip*/cuda* the same way the
// --- macros above do.  __PREFIX is still in scope here (only __DO_CUDA /
// --- __DO_HIP are undef'd, below).
#define gpuMalloc               CONCAT(__PREFIX, Malloc)
#define gpuMemcpy               CONCAT(__PREFIX, Memcpy)
#define gpuMemcpyAsync          CONCAT(__PREFIX, MemcpyAsync)
#define gpuMemset               CONCAT(__PREFIX, Memset)
#define gpuMemsetAsync          CONCAT(__PREFIX, MemsetAsync)
#define gpuMemcpyHostToDevice   CONCAT(__PREFIX, MemcpyHostToDevice)
#define gpuMemcpyDeviceToHost   CONCAT(__PREFIX, MemcpyDeviceToHost)
#define gpuMemcpyDeviceToDevice CONCAT(__PREFIX, MemcpyDeviceToDevice)
#define gpuStream_t             CONCAT(__PREFIX, Stream_t)
#define gpuStreamCreate         CONCAT(__PREFIX, StreamCreate)
#define gpuStreamDestroy        CONCAT(__PREFIX, StreamDestroy)
#define gpuStreamSynchronize    CONCAT(__PREFIX, StreamSynchronize)

// Pinned host allocation has different spellings between vendors, so it is
// not a simple CONCAT.  Provide small inline wrappers instead.
#if defined __DO_HIP
   static inline void gpuHostAllocPinned(void** p, size_t bytes) { hipHostMalloc(p, bytes); }
   static inline void gpuHostFreePinned (void*  p)               { hipHostFree(p); }
#elif defined __DO_CUDA
   static inline void gpuHostAllocPinned(void** p, size_t bytes) { cudaMallocHost(p, bytes); }
   static inline void gpuHostFreePinned (void*  p)               { cudaFreeHost(p); }
#endif


#undef __DO_CUDA
#undef __DO_HIP

#endif // #ifndef GPUPORTABILITY_HH
