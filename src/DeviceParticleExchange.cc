#include "DeviceParticleExchange.hh"

#ifdef GPU_COLLECTIVES

#include "ParticleVault.hh"
#include "ParticleVaultContainer.hh"
#include "SendQueue.hh"
#include "MemoryControl.hh"
#include "cudaFunctions.hh"
#include "cudaUtils.hh"
#include "qs_assert.hh"
#include <cstdio>

//----------------------------------------------------------------------------
// Pack kernel: for every (neighbor, particleIndex) tuple in the send queue,
// atomically grab a slot in that neighbor's send slab and copy the particle in.
//----------------------------------------------------------------------------
__global__ void packParticlesKernel( SendQueue* sendQueue,
                                      ParticleVault* processingVault,
                                      MC_Base_Particle* d_sendBuf,
                                      int* d_sendCount,
                                      int maxPerRank )
{
    int i = getGlobalThreadID();
    if ( i >= sendQueue->sizeDevice() ) return;

    sendQueueTuple& t = sendQueue->getTupleDevice( i );
    int r = t._neighbor;

    int slot = atomicAdd( &d_sendCount[r], 1 );
    if ( slot < maxPerRank )
    {
        d_sendBuf[ r * maxPerRank + slot ] =
            processingVault->getBaseParticleRef( t._particleIndex );
    }
    // slot >= maxPerRank is caught as a hard error on the host (count overflow).
}

//----------------------------------------------------------------------------
// Unpack kernel: append the particles received from one source rank into the
// container's extra vaults (cleanExtraVaults() later folds them into processing).
//----------------------------------------------------------------------------
__global__ void unpackParticlesKernel( MC_Base_Particle* d_recvBuf,
                                        int rankBase,
                                        int count,
                                        ParticleVaultContainer* vaultContainer )
{
    int i = getGlobalThreadID();
    if ( i >= count ) return;

    MC_Base_Particle p = d_recvBuf[ rankBase + i ];
    p.last_event = MC_Tally_Event::Facet_Crossing_Communication;
    vaultContainer->addExtraBaseParticle( p );
}

//----------------------------------------------------------------------------
DeviceParticleExchange::DeviceParticleExchange( int nRanks, int maxParticlesPerRank )
: _nRanks(nRanks),
  _maxPerRank(maxParticlesPerRank),
  _myRank(g_gpuColl.rank),
  _d_sendBuf(NULL), _d_recvBuf(NULL),
  _d_sendCount(NULL), _h_sendCount(NULL), _h_recvCount(NULL)
{
    size_t slab = (size_t)_nRanks * (size_t)_maxPerRank;

    _d_sendBuf   = MemoryControl::allocate<MC_Base_Particle>( slab,    MemoryControl::AllocationPolicy::DEVICE_MEM );
    _d_recvBuf   = MemoryControl::allocate<MC_Base_Particle>( slab,    MemoryControl::AllocationPolicy::DEVICE_MEM );
    _d_sendCount = MemoryControl::allocate<int>(              _nRanks, MemoryControl::AllocationPolicy::DEVICE_MEM );
    _h_sendCount = MemoryControl::allocate<int>(              _nRanks, MemoryControl::AllocationPolicy::PINNED_HOST_MEM );
    _h_recvCount = MemoryControl::allocate<int>(              _nRanks, MemoryControl::AllocationPolicy::PINNED_HOST_MEM );
}

//----------------------------------------------------------------------------
DeviceParticleExchange::~DeviceParticleExchange()
{
    size_t slab = (size_t)_nRanks * (size_t)_maxPerRank;
    MemoryControl::deallocate( _d_sendBuf,   slab,    MemoryControl::AllocationPolicy::DEVICE_MEM );
    MemoryControl::deallocate( _d_recvBuf,   slab,    MemoryControl::AllocationPolicy::DEVICE_MEM );
    MemoryControl::deallocate( _d_sendCount, _nRanks, MemoryControl::AllocationPolicy::DEVICE_MEM );
    MemoryControl::deallocate( _h_sendCount, _nRanks, MemoryControl::AllocationPolicy::PINNED_HOST_MEM );
    MemoryControl::deallocate( _h_recvCount, _nRanks, MemoryControl::AllocationPolicy::PINNED_HOST_MEM );
}

//----------------------------------------------------------------------------
void DeviceParticleExchange::beginRound()
{
    gpuMemset( _d_sendCount, 0, _nRanks * sizeof(int) );
    gpuDeviceSynchronize();
}

//----------------------------------------------------------------------------
void DeviceParticleExchange::pack( SendQueue* sendQueue, ParticleVault* processingVault )
{
    int n = sendQueue->size();
    if ( n <= 0 ) return;

    dim3 grid(1,1,1), block(1,1,1);
    int run = ThreadBlockLayout( grid, block, n );
    if ( run )
    {
        packParticlesKernel<<<grid, block>>>( sendQueue, processingVault,
                                              _d_sendBuf, _d_sendCount, _maxPerRank );
        gpuPeekAtLastError();
        gpuDeviceSynchronize();
    }
}

//----------------------------------------------------------------------------
uint64_t DeviceParticleExchange::exchange( ParticleVaultContainer* vaultContainer )
{
    // 1) Pull per-rank send counts to the host.
    MemoryControl::copyToHost( _h_sendCount, _d_sendCount, _nRanks );

    // Hard error if any destination overflowed its fixed-capacity slab.
    for ( int r = 0; r < _nRanks; ++r )
        qs_assert( _h_sendCount[r] <= _maxPerRank );

    // 2) All-to-all of the counts (control plane only - tiny).
    MPI_Alltoall( _h_sendCount, 1, MPI_INT,
                  _h_recvCount, 1, MPI_INT, g_gpuColl.mpiComm );

    for ( int r = 0; r < _nRanks; ++r )
        qs_assert( _h_recvCount[r] <= _maxPerRank );

    // 3) All-to-all-v of the payloads via grouped RCCL/NCCL P2P.
    //    Particles travel as raw bytes (MC_Base_Particle is trivially copyable).
    const size_t pbytes = sizeof(MC_Base_Particle);

    ncclGroupStart();
    for ( int r = 0; r < _nRanks; ++r )
    {
        if ( r == _myRank ) continue;
        if ( _h_sendCount[r] > 0 )
            ncclSend( _d_sendBuf + (size_t)r * _maxPerRank,
                      (size_t)_h_sendCount[r] * pbytes, ncclChar,
                      r, g_gpuColl.comm, g_gpuColl.stream );
        if ( _h_recvCount[r] > 0 )
            ncclRecv( _d_recvBuf + (size_t)r * _maxPerRank,
                      (size_t)_h_recvCount[r] * pbytes, ncclChar,
                      r, g_gpuColl.comm, g_gpuColl.stream );
    }
    ncclGroupEnd();
    gpuStreamSynchronize( g_gpuColl.stream );

    // 4) Append received particles into the extra vaults on the device.
    uint64_t totalRecv = 0;
    for ( int r = 0; r < _nRanks; ++r )
    {
        int c = _h_recvCount[r];
        if ( c <= 0 ) continue;
        totalRecv += (uint64_t)c;

        dim3 grid(1,1,1), block(1,1,1);
        int run = ThreadBlockLayout( grid, block, c );
        if ( run )
            unpackParticlesKernel<<<grid, block, 0, g_gpuColl.stream>>>(
                _d_recvBuf, (int)((size_t)r * _maxPerRank), c, vaultContainer );
    }
    gpuStreamSynchronize( g_gpuColl.stream );
    gpuPeekAtLastError();

    return totalRecv;
}

#endif // GPU_COLLECTIVES
