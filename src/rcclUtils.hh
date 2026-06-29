#ifndef RCCL_UTILS_HH
#define RCCL_UTILS_HH

#include "gpuPortability.hh"

// Pick the GPU collectives backend.  RCCL (AMD) and NCCL (NVIDIA) share the
// same ncclXxx API spellings, so only the header differs.  Either one defines
// GPU_COLLECTIVES, which gates every GPU-resident communication path.
#if defined(HAVE_RCCL)
   #include <rccl/rccl.h>
   #define GPU_COLLECTIVES
#elif defined(HAVE_NCCL)
   #include <nccl.h>
   #define GPU_COLLECTIVES
#endif

#ifdef GPU_COLLECTIVES

#include "utilsMpi.hh"
#include <cstdint>

//----------------------------------------------------------------------------
// Process-wide GPU collectives context.  One GPU per MPI rank is assumed
// (the launcher pins each rank to a device, or we fall back to rank % count).
//----------------------------------------------------------------------------
struct GpuCollectives
{
    ncclComm_t  comm;
    gpuStream_t stream;
    MPI_Comm    mpiComm;     // used for the count-exchange control plane
    int         rank;
    int         nRanks;
    bool        initialized;
};

extern GpuCollectives g_gpuColl;

// Bootstrap RCCL/NCCL from an existing MPI communicator.  Must be called after
// mpiInit and before any collective.  Broadcasts the ncclUniqueId over MPI.
void rcclInit( MPI_Comm mpiComm, int rank, int nRanks );

// Destroy the communicator and stream.  Call before mpiFinalize.
void rcclFinalize();

// Blocking sum-allreduce helpers that take/return *host* buffers.  They stage
// through a small device scratch buffer internally so callers can keep working
// with host-side counters (done-test, balance tallies).
void rcclAllReduceSumInt64 ( const int64_t*  sendHost, int64_t*  recvHost, int count );
void rcclAllReduceSumUint64( const uint64_t* sendHost, uint64_t* recvHost, int count );

#endif // GPU_COLLECTIVES
#endif // RCCL_UTILS_HH
