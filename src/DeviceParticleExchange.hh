#ifndef DEVICE_PARTICLE_EXCHANGE_HH
#define DEVICE_PARTICLE_EXCHANGE_HH

#include "rcclUtils.hh"

#ifdef GPU_COLLECTIVES

#include "MC_Base_Particle.hh"
#include <cstdint>

class ParticleVault;
class ParticleVaultContainer;
class SendQueue;

//----------------------------------------------------------------------------
// DeviceParticleExchange
//
// GPU-resident replacement for the host MPI_Isend/Irecv particle migration in
// cycleTracking.  It keeps particle payloads in explicit device memory the
// whole way through:
//
//   pack kernel  : SendQueue (+ processing vault) -> per-destination device
//                  send buffers, grouped by rank.
//   count xchg   : per-rank send counts exchanged via MPI_Alltoall (tiny
//                  control-plane message; payloads never touch the host).
//   payload xchg : RCCL/NCCL all-to-all-v, implemented as grouped ncclSend /
//                  ncclRecv (NCCL has no native variadic all-to-all). Only
//                  nonzero-count peers post a transfer, so sparse neighbour
//                  traffic stays efficient.
//   unpack kernel: received device buffers -> the container's extra vaults,
//                  which cleanExtraVaults() then folds into processing.
//
// Capacities are fixed at construction (one GPU-friendly contiguous slab per
// direction). Overflow is a hard error, matching the fixed-capacity design.
//----------------------------------------------------------------------------
class DeviceParticleExchange
{
  public:
    // maxParticlesPerRank bounds how many particles this rank may send to (or
    // receive from) any single other rank in one comm round.
    DeviceParticleExchange( int nRanks, int maxParticlesPerRank );
    ~DeviceParticleExchange();

    // Zero the per-rank send counters. Call once at the start of a comm round
    // (i.e. before tracking the cycle's processing vaults).
    void beginRound();

    // Pack one processing vault's off-rank particles (recorded in sendQueue)
    // into the device send slabs. May be called repeatedly within a round to
    // accumulate across multiple processing vaults before the exchange.
    void pack( SendQueue* sendQueue, ParticleVault* processingVault );

    // Exchange counts + payloads and append received particles into the
    // container's extra vaults. Returns the total particles received.
    uint64_t exchange( ParticleVaultContainer* vaultContainer );

  private:
    int _nRanks;
    int _maxPerRank;
    int _myRank;

    MC_Base_Particle* _d_sendBuf;    // device  [_nRanks * _maxPerRank]
    MC_Base_Particle* _d_recvBuf;    // device  [_nRanks * _maxPerRank]
    int*              _d_sendCount;  // device  [_nRanks]
    int*              _h_sendCount;  // pinned host [_nRanks]
    int*              _h_recvCount;  // pinned host [_nRanks]

    // disable copy
    DeviceParticleExchange( const DeviceParticleExchange& );
    DeviceParticleExchange& operator=( const DeviceParticleExchange& );
};

#endif // GPU_COLLECTIVES
#endif // DEVICE_PARTICLE_EXCHANGE_HH
