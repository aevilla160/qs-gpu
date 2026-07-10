Let me quickly check if you have the MSCCL++ source locally so I can ground this in the exact API version you're building against.

Looked for local mscclpp checkout
You have the MSCCL++ source locally — let me check exactly which device primitives your version offers.


Ran 2 commands
Yes — several, and your checkout of the source confirms what's available. The important part: ranked against your own profiling data, the biggest MSCCL++ intra-node lever isn't the payload path you've already built — it's attacking the small-message collectives that your done_test/count instrumentation just exposed as the floor. Here's the menu, grounded in the API in /Users/avilla/Downloads/softwares/mscclpp/include/mscclpp/.

What you use today (baseline)
MemoryChannel::put<4,true> + signal/wait + a relaxedSignal/relaxedWait pre-barrier, one block per peer, host-driven round loop. At 4 GPUs this payload path costs 33 ms cumulative — already ~2% of comm. Optimizing it further is polishing the wrong thing.

Ranked opportunities
1. Kernel-initiated tiny collectives — replace the RCCL count all-to-all. Your floor item: count_alltoall = 0.61 s at 4 GPUs, ~2.6 ms/call, for 4 ints. MemoryChannel exposes raw mapped-peer access — write<T>(index, v) / read<T>(index) (memory_channel_device.hpp:74–86) — because CudaIpc maps the peer's buffer into your address space over XGMI. A 4-rank alltoall of one int is literally three remote 4-byte stores + a flag poll: ~1–2 µs, not 2.6 ms. That's a ~1000× latency gap on the measured bottleneck. This is also the intellectually honest extension of your research sub-question: "kernel-initiated primitives vs RCCL in the small-message tail" applies more to the control plane than to the payload.

2. Fuse count + payload into one one-sided operation (eliminates the count phase entirely). The count exchange exists only so receivers know how many bytes are coming. With one-sided puts you can carry that in-band: put the payload, then write a small header (count, round-id) per peer via an LL16Packet (packet_device.hpp — write(uint64_t val, uint32_t flag) / spin-read(flag)). Use the round number as the packet flag and you never zero buffers. Receiver polls 1 header per peer, then unpacks. This removes the count all-to-all and the count D2H copy and one stream sync per round. At 4 GPUs that's ~0.61 s off a 0.84 s comm total → comm drops ~73%, wall drops ~38% (1.61 s → ~1.0 s). Nothing on the RCCL side can do this — it's structurally unique to one-sided.

3. LL packets for the sparse payload tail. putPackets/unpackPackets (memory_channel_device.hpp:155–218) send data+flag in a single store — no separate signal/wait round trip, and no per-round all-peers handshake barrier (packet flags self-synchronize). 2× bandwidth cost, irrelevant for the <64 KB rounds that make up half your tail. Dispatch by size: bytes < threshold → packets, else put+signal.

4. Zero-copy direct pack. Because the peer's recv slab is directly addressable, the pack kernel could write particles straight into the remote slab (via write<T> at an atomically-reserved slot) instead of pack → local slab → put. Merges two device passes into one, halves slab memory, removes the put kernel launch.

5. GPU-driven persistent kernel (your port doc's stated end-state). Move the round loop device-side: track → pack → exchange (packets) → unpack → device-side done-test allreduce, in one persistent kernel. Kills ~1000+ kernel launches and per-round host syncs per run. Big restructure — do it after 1–3 prove out.

6. (Minor) PortChannel over CudaIpc for SDMA offload. The proxy path isn't IB-only; it can drive DMA-engine copies intra-node, freeing CUs during the 12 MB round-0 transfers. Your code currently only builds port channels when IB exists — would need a small change. Low priority: payload isn't the wall.

Not applicable / not the target
SwitchChannel (switch_channel*.hpp) — NVLS multicast, NVSwitch/NVIDIA-only. Dead on MI300A.
IB PortChannel inter-node — no verbs on Slingshot-11 (already scoped out).
Executor/algorithm DSL — for standard collectives from JSON plans; your exchange is custom, raw channels are the right level. (Upstream also ships an NCCL-API interposer under apps/nccl, but your checkout doesn't have it built — building it would let you A/B RCCL vs MSCCL++ allreduce with zero code change, worth knowing about.)
Caveats
LL packet correctness on AMD depends on 16-byte store atomicity/visibility over XGMI. Validate with the QS_MSCCLPP_VERIFY harness we just built — it's backend-agnostic and will catch drops/corruption from any of these changes.
MI300A is an APU (CPU/GPU coherent memory), so the D2H count copy was never the cost (0.7 ms total); it's RCCL's collective latency you're beating. Frame it that way in the writeup.
Recommendation
Microbenchmark first, integrate second: a standalone 4-rank test comparing (a) RCCL alltoall of 1 int vs MemoryChannel::write+flag, and (b) RCCL allreduce of 2×int64 vs an LL-packet allreduce. That bounds the win in one afternoon. Then implement #2 (fused header), since it subsumes #1 for the count path and directly removes the measured 0.61 s. Want me to sketch the fused header-packet exchange design against your current DeviceParticleExchangeMscclpp.cc structure?

