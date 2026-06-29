#include "rcclUtils.hh"

#ifdef GPU_COLLECTIVES

#include "qs_assert.hh"
#include <cstdio>

GpuCollectives g_gpuColl = { };

#define RCCL_CHECK(call)                                                       \
   do {                                                                        \
      ncclResult_t _e = (call);                                               \
      if (_e != ncclSuccess) {                                                 \
         fprintf(stderr, "RCCL/NCCL error '%s' at %s:%d\n",                    \
                 ncclGetErrorString(_e), __FILE__, __LINE__);                  \
         qs_assert(false);                                                     \
      }                                                                        \
   } while (0)

//----------------------------------------------------------------------------
void rcclInit( MPI_Comm mpiComm, int rank, int nRanks )
{
    // Bind this rank to a GPU (one GPU per rank).  If the launcher already
    // restricts visible devices to one per rank, this is a no-op (count==1).
    int deviceCount = 0;
    gpuGetDeviceCount(&deviceCount);
    if (deviceCount > 0)
        gpuSetDevice( rank % deviceCount );

    // Rank 0 generates the unique id; broadcast it to everyone over MPI.
    ncclUniqueId id;
    if (rank == 0)
        RCCL_CHECK( ncclGetUniqueId(&id) );
    mpiBcast(&id, sizeof(id), MPI_BYTE, 0, mpiComm);

    RCCL_CHECK( ncclCommInitRank(&g_gpuColl.comm, nRanks, id, rank) );
    gpuStreamCreate(&g_gpuColl.stream);

    g_gpuColl.mpiComm     = mpiComm;
    g_gpuColl.rank        = rank;
    g_gpuColl.nRanks      = nRanks;
    g_gpuColl.initialized = true;
}

//----------------------------------------------------------------------------
void rcclFinalize()
{
    if (!g_gpuColl.initialized) return;
    gpuStreamSynchronize(g_gpuColl.stream);
    ncclCommDestroy(g_gpuColl.comm);
    gpuStreamDestroy(g_gpuColl.stream);
    g_gpuColl.initialized = false;
}

//----------------------------------------------------------------------------
void rcclAllReduceSumInt64( const int64_t* sendHost, int64_t* recvHost, int count )
{
    int64_t *d_send = NULL, *d_recv = NULL;
    size_t bytes = (size_t)count * sizeof(int64_t);

    gpuMalloc((void**)&d_send, bytes);
    gpuMalloc((void**)&d_recv, bytes);
    gpuMemcpy(d_send, sendHost, bytes, gpuMemcpyHostToDevice);

    RCCL_CHECK( ncclAllReduce(d_send, d_recv, count, ncclInt64, ncclSum,
                              g_gpuColl.comm, g_gpuColl.stream) );
    gpuStreamSynchronize(g_gpuColl.stream);

    gpuMemcpy(recvHost, d_recv, bytes, gpuMemcpyDeviceToHost);
    gpuFree(d_send);
    gpuFree(d_recv);
}

//----------------------------------------------------------------------------
void rcclAllReduceSumUint64( const uint64_t* sendHost, uint64_t* recvHost, int count )
{
    uint64_t *d_send = NULL, *d_recv = NULL;
    size_t bytes = (size_t)count * sizeof(uint64_t);

    gpuMalloc((void**)&d_send, bytes);
    gpuMalloc((void**)&d_recv, bytes);
    gpuMemcpy(d_send, sendHost, bytes, gpuMemcpyHostToDevice);

    RCCL_CHECK( ncclAllReduce(d_send, d_recv, count, ncclUint64, ncclSum,
                              g_gpuColl.comm, g_gpuColl.stream) );
    gpuStreamSynchronize(g_gpuColl.stream);

    gpuMemcpy(recvHost, d_recv, bytes, gpuMemcpyDeviceToHost);
    gpuFree(d_send);
    gpuFree(d_recv);
}

#endif // GPU_COLLECTIVES
