# GPU Port Survey: Quicksilver HIP Residency and RCCL/NCCL Communication

## Scope

This note surveys the current Quicksilver code in this directory and outlines what must change to make the application GPU portable with HIP, fully GPU resident in the steady state, and able to use RCCL/NCCL collectives or P2P communication for GPU buffers.

No source changes are proposed here as code. This is a porting plan and risk map.

## Working Assumptions

- Target accelerator path is HIP on AMD GPUs first.
- "Fully resident" means particle vaults, communication buffers, tally buffers, reduction inputs, and migrated particle payloads should live in explicit device memory during cycle tracking.
- UVM is not sufficient for the desired target, even when it avoids source changes.
- MPI may still be used for bootstrap/control-plane work such as argument parsing, rank setup, communicator ID exchange, and final host-side output.
- RCCL is the AMD target library. It uses NCCL-style API names such as `ncclAllReduce`, `ncclSend`, and `ncclRecv`.
- NCCL is relevant as the API model and as the NVIDIA backend if the code remains portable across vendors.

## Executive Summary

Quicksilver already has a partial HIP build path, but it is not a fully GPU-resident design. The current implementation is best described as:

1. Host initializes mesh, nuclear data, material data, particle vaults, tallies, and MPI state.
2. HIP/CUDA-capable code launches a GPU transport kernel.
3. The code synchronizes the GPU after each vault.
4. Host code walks a device-callable send queue, packs particle data into host byte buffers, sends with MPI, receives with MPI, unpacks on the host, collapses vaults, and performs done-test reductions.

The transport kernel is real. The residency problem is the infrastructure around it.

The most important conclusion: replacing CUDA spellings with HIP spellings is not the hard part. The hard part is replacing host-side particle communication, host-side vault compaction, host-side source/population control, and host-side reductions with explicit device-memory paths.

## Codebase Scale

Approximate survey counts:

| Category | Count |
| --- | ---: |
| C++ source/header files | 117 |
| C++ source/header lines | 13,118 |
| GPU/UVM-related hits | 86 |
| MPI wrapper/direct MPI hits | 126 |
| Host container/allocation/free hits | 231 |

These counts are directional, not a formal metric. They are useful mainly because they show the port is not confined to one kernel file.

## Current HIP/GPU State

### HIP Build Path Exists

The Makefile has an AMD HIP configuration:

- `src/Makefile:110`
- `CPPFLAGS = -DHAVE_MPI -DHAVE_HIP -x hip --offload-arch=gfx90a -fgpu-rdc -Wno-unused-result`
- `LDFLAGS = -fgpu-rdc --hip-link --offload-arch=gfx90a`

This is a useful starting point, but it is hardcoded to a specific LLNL Cray/ROCm environment. A portable build should move this into a selectable build profile or a CMake/HIP build configuration.

### GPU Portability Layer Assumes UVM

`src/gpuPortability.hh` maps both CUDA and HIP builds to `HAVE_UVM`:

- `src/gpuPortability.hh:11` selects HIP.
- `src/gpuPortability.hh:14` defines `HAVE_UVM`.
- `src/gpuPortability.hh:32` maps `VAR_MEM` to `MemoryControl::AllocationPolicy::UVM_MEM`.

`src/MemoryControl.hh` then allocates `UVM_MEM` with `gpuMallocManaged`:

- `src/MemoryControl.hh:23`
- `src/MemoryControl.hh:25`

This directly conflicts with explicit full residency. A full-residency port needs a new explicit device allocation mode, probably `DEVICE_MEM`, using `hipMalloc` and `hipFree`.

### GPU Transport Kernel Exists

The native GPU transport entry point is in `src/main.cc`:

- `CycleTrackingKernel` is defined at `src/main.cc:126`.
- It calls `CycleTrackingGuts`.
- The kernel is launched at `src/main.cc:193`.

After launch, the code immediately calls:

- `gpuPeekAtLastError()`
- `gpuDeviceSynchronize()`

at `src/main.cc:196-197`.

That synchronization makes sense for the current host communication path, but it prevents an asynchronous GPU-resident pipeline.

## Current Data Residency

### `qs_vector`

`qs_vector` is the central custom container:

- Device-qualified indexing exists at `src/QS_Vector.hh:85`.
- Device-qualified `size()` exists at `src/QS_Vector.hh:103`.
- Device atomic size increment exists at `src/QS_Vector.hh:178`.

But key operations are host-oriented:

- Constructors allocate through `MemoryControl`.
- `reserve`, `resize`, `push_back`, `clear`, and `appendList` are host-side container operations except for direct indexing and atomic increment.
- Copy construction copies element-by-element on the host.
- Destructor deallocates through `MemoryControl`.

This is a workable transitional container for UVM. It is not yet a clean explicit device container.

### Particle Vaults

`ParticleVault` stores `qs_vector<MC_Base_Particle>`. It has device-callable particle push, get, and invalidate functions. That is favorable.

The problem is `ParticleVaultContainer`:

- `_processingVault` and `_processedVault` are `std::vector<ParticleVault*>` at `src/ParticleVaultContainer.hh:104` and `src/ParticleVaultContainer.hh:107`.
- `_extraVault` is `qs_vector<ParticleVault*>` at `src/ParticleVaultContainer.hh:110`.
- `collapseProcessing`, `collapseProcessed`, `swapProcessingProcessedVaults`, `addProcessingParticle`, and `cleanExtraVaults` are host procedures.

This means the current vault topology is not a GPU-resident dynamic structure. It is a host-managed list of UVM/device-accessible particle vault objects.

## Current Communication Path

### Off-Rank Detection Is Device-Callable

When a particle crosses off-process, the GPU-capable path does this:

- Computes neighbor rank at `src/MC_Facet_Crossing_Event.cc:60`.
- Writes updated particle state back to the processing vault at `src/MC_Facet_Crossing_Event.cc:62`.
- Pushes `(neighbor_rank, particle_index)` into `SendQueue` at `src/MC_Facet_Crossing_Event.cc:65`.

This is a good hook. It is the point where device-side communication packing should begin.

### Host Takes Over Immediately After Kernel

After each transport kernel, `src/main.cc`:

- Gets `SendQueue` at `src/main.cc:249`.
- Allocates send buffers at `src/main.cc:250`.
- Host-walks `sendQueue` at `src/main.cc:253`.
- Gets particle data from the processing vault at `src/main.cc:258`.
- Serializes into `MC_Particle_Buffer` at `src/main.cc:261`.
- Sends buffers at `src/main.cc:264`.
- Clears vault and send queue at `src/main.cc:266-267`.
- Cleans extra vaults at `src/main.cc:270`.
- Receives MPI particle buffers at `src/main.cc:273`.
- Collapses vaults at `src/main.cc:282-283`.
- Tests completion at `src/main.cc:289`.

This block is the primary residency barrier.

### `MC_Particle_Buffer` Uses Host Buffers

`src/MC_Particle_Buffer.cc` allocates byte buffers with host allocation macros:

- `particle_buffer_base_type::Allocate` at `src/MC_Particle_Buffer.cc:59`
- `MC_MALLOC` at `src/MC_Particle_Buffer.cc:72`

It serializes particles into separate int, double, and char regions:

- `Buffer_Particle` at `src/MC_Particle_Buffer.cc:424`
- `MC_Base_Particle::Serialize` is used for packing.

It communicates with MPI:

- `mpiIsend` at `src/MC_Particle_Buffer.cc:495`
- `mpiIrecv` at `src/MC_Particle_Buffer.cc:518`
- receive repost at `src/MC_Particle_Buffer.cc:542`

It performs done-test reductions with MPI:

- blocking `mpiAllreduce` at `src/MC_Particle_Buffer.cc:608`
- nonblocking `mpiIAllreduce` at `src/MC_Particle_Buffer.cc:638`

This code must be replaced or significantly split for GPU-resident communication.

## Current Host-Side Cycle Work

Even if particle migration is moved to GPU buffers, other cycle work remains host-driven.

### Source Generation

`MC_SourceNow` is host-side:

- Uses `std::vector<double> source_rate` at `src/MC_SourceNow.cc:32`.
- Loops domains and cells on the host at `src/MC_SourceNow.cc:43`.
- Uses `mpiAllreduce` for source weight at `src/MC_SourceNow.cc:57`.
- Generates particles in host loops at `src/MC_SourceNow.cc:74-126`.

### Population Control

`PopulationControl` is host-side:

- Counts processing particles at `src/PopulationControl.cc:26`.
- Uses `mpiAllreduce` at `src/PopulationControl.cc:41`.
- Splits/roulettes particles with host loops at `src/PopulationControl.cc:71`.
- Kills particles with `eraseSwapParticle` at `src/PopulationControl.cc:92` and `src/PopulationControl.cc:164`.

### Tally Finalization

Tallies are incremented by GPU-capable atomics during tracking, but finalization is host-side:

- `SumTasks` at `src/Tallies.cc:16`
- `CycleFinalize` at `src/Tallies.cc:25`
- host `std::vector<uint64_t>` reduction buffers at `src/Tallies.cc:29`
- `mpiAllreduce` at `src/Tallies.cc:46`
- scalar flux sum host loops at `src/Tallies.cc:146`
- another `mpiAllreduce` at `src/Tallies.cc:168`

## RCCL/NCCL Fit

### Good Fit

RCCL/NCCL are a good fit for:

- global allreduce of done-test particle counts
- global allreduce of balance tallies
- scalar flux reductions if reduced into dense device buffers
- fixed-size dense summary reductions
- possibly allgather/alltoall metadata exchange for neighbor counts

Relevant docs:

- NCCL collective API: <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html>
- NCCL P2P API: <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html>
- RCCL documentation: <https://rocm.docs.amd.com/projects/rccl/en/latest/>
- RCCL library specification: <https://rocm.docs.amd.com/projects/rccl/en/latest/api-reference/library-specification.html>

### Not a Direct Fit

The current particle exchange is sparse, variable-count, neighbor-specific migration. It is currently implemented as asynchronous MPI point-to-point messages with per-neighbor byte buffers.

Plain collectives such as allreduce do not express this.

Options:

1. Use RCCL/NCCL P2P `ncclSend` and `ncclRecv`.
2. Use RCCL `ncclAllToAllv` if the target RCCL version and platform support it robustly.
3. Use GPU-aware MPI for particle migration and RCCL only for reductions.
4. Redesign particle migration into dense all-to-all exchange. This is simple conceptually but may be memory-inefficient for sparse neighbor traffic.

### P2P Caveat

NCCL P2P sends/receives require matching datatype and count between sender and receiver. Concurrent sends/receives that depend on each other must be grouped with `ncclGroupStart` and `ncclGroupEnd`.

That implies a two-phase communication protocol:

1. Exchange per-neighbor send counts.
2. Allocate/select receive slots.
3. Group all sends and receives.
4. Unpack or append received particles on the GPU.

## Recommended Architecture

### Memory Model

Add explicit memory policies:

```cpp
enum AllocationPolicy {
    HOST_MEM,
    DEVICE_MEM,
    MANAGED_MEM,
    PINNED_HOST_MEM,
    UNDEFINED_POLICY
};
```

For HIP:

- `DEVICE_MEM`: `hipMalloc`, `hipFree`
- `MANAGED_MEM`: `hipMallocManaged`, only as an optional compatibility path
- `PINNED_HOST_MEM`: `hipHostMalloc`, `hipHostFree`, only for staging/control paths

Do not keep mapping `VAR_MEM` to managed memory by default. Make it explicit:

- `STATE_MEM` for long-lived simulation state
- `COMM_MEM` for communication payloads
- `HOST_CONTROL_MEM` for rank/control state

### Particle Layout

The current byte-serialized int/double/char split is not ideal for RCCL/NCCL.

Recommended first GPU-resident format:

```cpp
struct DeviceParticlePacket {
    MC_Base_Particle particle;
};
```

Then send as `ncclUint8` or equivalent byte count if needed, or use a typed POD layout if the communication layer supports it cleanly.

Reasons:

- avoids device-side recreation of the current host serializer
- avoids three discontiguous payload regions
- keeps the first port leg simple
- preserves the current AoS particle model

Longer term, SoA may perform better for transport, but that is a separate optimization. Do not combine it with the first residency port unless profiling proves it necessary.

### Device-Side Send Pipeline

Replace the host queue walk in `src/main.cc:253` with kernels:

1. Count remote particles per neighbor from `SendQueue`.
2. Prefix-sum counts into per-neighbor offsets.
3. Pack `MC_Base_Particle` packets into device send buffers.
4. Clear or compact source vault metadata on device.
5. Exchange counts.
6. Exchange payloads with RCCL/NCCL P2P or GPU-aware MPI.
7. Append received particles to device processing vaults.

For a first implementation, use one HIP stream for tracking and one HIP stream for communication. Keep correctness first. Add overlap later.

### Reductions

Replace host `mpiAllreduce` for resident values with RCCL/NCCL where practical:

- done-test gains/losses: two `int64` values per rank
- balance tallies: 13 `uint64` values per rank
- scalar flux: reduce dense device buffer, not host nested loops

Keep a small host copy only when printing or deciding host-side loop control. If strict residency is required, loop continuation should be based on device-side flags copied only after a reduction completes, or use a minimal scalar copy.

## Porting Milestones

### Milestone 0: Build and Baseline

Goal: establish a reproducible HIP baseline before changing semantics.

Tasks:

- Parameterize the HIP build instead of hardcoding LLNL paths.
- Build with `HAVE_HIP`.
- Run a small single-rank input.
- Run a small multi-rank input if the target machine is available.
- Record correctness outputs and timing.

Deliverable:

- Baseline command lines.
- Known-good input file.
- Expected tally summary.

### Milestone 1: Explicit Device Allocation

Goal: remove managed memory from the hot-state path.

Tasks:

- Add explicit `DEVICE_MEM`.
- Convert particle vault storage to `DEVICE_MEM`.
- Convert mesh/nuclear/material read-mostly data to `DEVICE_MEM` or explicit copied device mirrors.
- Keep host mirrors only for initialization and output.
- Add device/host copy utilities with explicit names.

Risks:

- Constructors and destructors currently assume host execution.
- `qs_vector` copy/assignment paths are host loops.
- `BulkStorage` uses host reference counting.

Validation:

- Single-rank CPU vs HIP tally comparison.
- HIP memcheck or sanitizing equivalent where available.
- No UVM allocations in transport state.

### Milestone 2: GPU Pack and Unpack

Goal: eliminate host walking of `SendQueue`.

Tasks:

- Keep `SendQueue` as the kernel-produced remote-crossing list.
- Add device counts per neighbor.
- Add device prefix-sum.
- Add device packet packing from particle vaults to send buffers.
- Add device append of received packets into processing vaults.

Risks:

- Dynamic vault growth currently uses host `std::vector`.
- Overflow policy for extra vaults is weakly guarded.
- Current `sendQueue.neighbor_size()` is a host linear scan.

Validation:

- Compare number of off-rank particles sent per neighbor against current MPI implementation.
- Compare final balance tallies on deterministic small cases.

### Milestone 3: GPU-Resident Communication

Goal: send and receive GPU buffers without host staging.

Preferred options in order:

1. GPU-aware MPI for particle P2P, RCCL for reductions.
2. RCCL/NCCL P2P for particle P2P with grouped send/recv calls.
3. RCCL all-to-all/all-to-all-v style exchange if available and efficient.

Harsh critique: "full use NCCL/RCCL collectives" is too broad for particle migration as currently designed. The particle exchange pattern is not a pure collective reduction. It is irregular sparse neighbor exchange. Forcing it through dense collectives may waste bandwidth and memory. Use RCCL/NCCL collectives where the data is actually collective: reductions and possibly count exchange. Use P2P or GPU-aware MPI for particle payloads unless a benchmark proves all-to-all is better.

Validation:

- Count exchange sanity by neighbor.
- Payload integrity checks using particle identifiers.
- Multi-rank balance invariants.
- Compare communication bytes and time to MPI baseline.

### Milestone 4: GPU Cycle Control

Goal: remove remaining host-side cycle work from the steady state.

Tasks:

- Port source generation to kernels.
- Port population control to kernels.
- Port low-weight roulette to kernels.
- Port vault collapse/compaction to kernels.
- Port tally local reductions to kernels.
- Use RCCL/NCCL for global reductions.

Risks:

- Vault compaction is currently serial and pointer-heavy.
- Random number stream reproducibility may change.
- Tally reduction order changes can affect floating-point reproducibility.

Validation:

- Statistical validation, not bitwise equality, for stochastic outputs.
- Balance invariants must remain exact for integer counters.
- Regression suite across example inputs.

## Suggested File-Level Worklist

| File | Porting Role | Required Change |
| --- | --- | --- |
| `src/gpuPortability.hh` | GPU API mapping | Stop equating HIP/CUDA with UVM. Add HIP stream and RCCL/NCCL wrappers. |
| `src/MemoryControl.hh` | Allocation policy | Add explicit device and pinned policies. |
| `src/QS_Vector.hh` | Core container | Separate host construction from device data access. Avoid hidden UVM reliance. |
| `src/ParticleVault.hh` | Particle storage | Preserve device push/get/invalidate. Add explicit capacity/overflow checks. |
| `src/ParticleVaultContainer.hh/.cc` | Vault topology | Replace host `std::vector` hot-path management with fixed or device-managed arrays. |
| `src/SendQueue.hh/.cc` | Remote crossing list | Add device count and pack support. Avoid host `neighbor_size()` scans. |
| `src/MC_Particle_Buffer.hh/.cc` | Current MPI particle transport | Either replace with `DeviceParticleExchange` or split host MPI and device exchange implementations. |
| `src/main.cc` | Cycle orchestration | Remove post-kernel host pack/comm/unpack block from the hot path. |
| `src/Tallies.cc` | Reductions and finalization | Move local reductions to GPU and global reductions to RCCL/NCCL. |
| `src/MC_SourceNow.cc` | Particle source | Port source generation to kernels. |
| `src/PopulationControl.cc` | Split/roulette | Port to kernels or make it an explicit host fallback. |

## Communication Design Sketch

### Current Flow

```text
GPU tracking kernel
  -> device-callable SendQueue
  -> synchronize
  -> host scans SendQueue
  -> host packs byte buffers
  -> MPI_Isend/Irecv host buffers
  -> host unpacks received particles
  -> host appends particles to vaults
```

### Target Flow

```text
HIP tracking kernel
  -> device SendQueue
  -> HIP count kernel by neighbor
  -> device prefix sum
  -> HIP pack kernel to device send buffers
  -> RCCL/NCCL or GPU-aware MPI count exchange
  -> RCCL/NCCL P2P or GPU-aware MPI payload exchange
  -> HIP unpack/append kernel
  -> RCCL/NCCL done-test allreduce
```

## Testing Strategy

### Correctness Tests

Use existing example inputs:

- `Examples/AllAbsorb/allAbsorb.inp`
- `Examples/AllEscape/allEscape.inp`
- `Examples/AllScattering/scatteringOnly.inp`
- `Examples/NoFission/noFission.inp`
- `Examples/Homogeneous/homogeneousProblem.inp`

For each:

- single rank CPU baseline
- single rank HIP baseline
- multi-rank MPI baseline
- multi-rank HIP resident communication path

### Invariants

Track these every cycle:

- start + source + produce + split equals absorb + census + escape + rr + fission when done
- off-rank send count equals matching remote receive count by neighbor pair
- no particle has invalid species in active vaults
- `SendQueue` size returns to zero after communication
- extra vault index returns to zero after cleanup

### Performance Measurements

Measure separately:

- tracking kernel time
- pack count and prefix time
- pack/unpack time
- count exchange time
- payload exchange time
- done-test reduction time
- forced synchronizations per cycle
- UVM page faults, if any managed memory remains

## Main Risks

1. Hidden UVM reliance.
   The code relies on managed pointers being visible to both host and device. Explicit memory will surface many ownership bugs.

2. Host-managed dynamic topology.
   `std::vector<ParticleVault*>` in the particle vault container is a major mismatch for GPU-resident execution.

3. Irregular particle communication.
   The communication pattern is sparse and variable-count. RCCL/NCCL collectives are excellent for dense reductions, but particle migration needs P2P or all-to-all-v style logic.

4. Serialization format.
   The current int/double/char serializer is host-friendly, not GPU-communication-friendly.

5. Reproducibility.
   Moving source generation, population control, reductions, and compaction to GPU will change operation order. Exact bitwise matching may not be feasible for all floating-point tallies.

## Strong Recommendation

Do not begin by rewriting the physics or changing particle layout to SoA. That would combine too many risks.

Start with this narrow sequence:

1. Keep `MC_Base_Particle` AoS.
2. Add explicit HIP device allocation.
3. Preserve the current tracking kernel.
4. Replace host send packing with device packet packing.
5. Use RCCL/NCCL allreduce for small global counters and tallies.
6. Use GPU-aware MPI or RCCL/NCCL P2P for particle payloads.
7. Only after correctness is stable, consider SoA particle storage and deeper transport-kernel optimization.

This keeps the first port focused on residency and communication, which is where the current design actually conflicts with the goal.

## Open Questions

1. Is GPU-aware MPI allowed as a transitional or final particle-migration backend, or must all inter-rank GPU payload traffic go through RCCL/NCCL?
2. Is the target strictly AMD HIP/RCCL, or should the abstraction preserve CUDA/NCCL as a first-class backend?
3. Is exact reproducibility required, or are statistical equivalence and integer balance invariants sufficient?
4. What target hardware and ROCm/RCCL versions should drive the design?
5. Is one GPU per MPI rank guaranteed?
6. Are dynamic vault growth and unbounded fission secondaries acceptable, or can the GPU path enforce fixed preallocated capacities?

