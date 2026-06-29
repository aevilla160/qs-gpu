#ifndef MEMORY_CONTROL_HH
#define MEMORY_CONTROL_HH

#include "gpuPortability.hh"
#include "qs_assert.hh"

namespace MemoryControl
{
   // HOST_MEM       : plain host new[]/delete[]
   // UVM_MEM        : managed memory (host+device visible) - transitional path
   // DEVICE_MEM     : explicit device memory (hipMalloc/hipFree) - no host access
   // PINNED_HOST_MEM: page-locked host memory for fast staging / async copies
   enum AllocationPolicy {HOST_MEM, UVM_MEM, DEVICE_MEM, PINNED_HOST_MEM, UNDEFINED_POLICY};

   template <typename T>
   T* allocate(const int size, const AllocationPolicy policy)
   {
      if (size == 0) { return NULL;}
      T* tmp = NULL;

      switch (policy)
      {
        case AllocationPolicy::HOST_MEM:
         tmp = new T [size];
         break;
#ifdef HAVE_UVM
        case AllocationPolicy::UVM_MEM:
         void *ptr;
         gpuMallocManaged(&ptr, size*sizeof(T));
         tmp = new(ptr) T[size];
         break;
#endif
#if defined GPU_NATIVE
        case AllocationPolicy::DEVICE_MEM:
         // Raw device memory.  No constructors are run and the pointer is NOT
         // host-dereferenceable; intended for POD payload/staging buffers that
         // are produced and consumed by kernels (e.g. communication buffers).
         gpuMalloc((void**)&tmp, size*sizeof(T));
         break;
        case AllocationPolicy::PINNED_HOST_MEM:
         gpuHostAllocPinned((void**)&tmp, size*sizeof(T));
         break;
#endif
      default:
         qs_assert(false);
         break;
      }
      return tmp;
   }

   template <typename T>
   void deallocate(T* data, const int size, const AllocationPolicy policy)
   {
      switch (policy)
      {
        case AllocationPolicy::HOST_MEM:
         delete[] data;
         break;
#ifdef HAVE_UVM
        case AllocationPolicy::UVM_MEM:
         for (int i=0; i < size; ++i)
            data[i].~T();
         gpuFree(data);
         break;
#endif
#if defined GPU_NATIVE
        case AllocationPolicy::DEVICE_MEM:
         gpuFree(data);
         break;
        case AllocationPolicy::PINNED_HOST_MEM:
         gpuHostFreePinned(data);
         break;
#endif
      default:
         qs_assert(false);
         break;
      }
   }

#if defined GPU_NATIVE
   // Explicit host<->device copy helpers for the DEVICE_MEM path.
   template <typename T>
   void copyToDevice(T* dst_device, const T* src_host, const int size)
   {
      gpuMemcpy(dst_device, src_host, size*sizeof(T), gpuMemcpyHostToDevice);
   }

   template <typename T>
   void copyToHost(T* dst_host, const T* src_device, const int size)
   {
      gpuMemcpy(dst_host, src_device, size*sizeof(T), gpuMemcpyDeviceToHost);
   }
#endif
}


#endif
