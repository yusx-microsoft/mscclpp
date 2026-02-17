// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// =============================================================================
// Companion CUDA kernel for Example 3: Memory Channel
// =============================================================================
// This kernel uses MemoryChannel to copy data from one GPU to another via
// direct memory-mapped access (NVLink/xGMI). The pattern is:
//   1. put() — copy data from local buffer to remote buffer
//   2. signal() + wait() — synchronize so both sides know the transfer is done
//
// The kernel is compiled at runtime by KernelBuilder in 3_memory_channel.py.
// =============================================================================

#include <mscclpp/memory_channel_device.hpp>

// We use extern "C" so KernelBuilder can find the kernel by name.
// The channel array has nranks entries; channels[my_rank] is unused (no self-send).
extern "C" __global__ void __launch_bounds__(1024, 1)
    memory_channel_put(mscclpp::MemoryChannelDeviceHandle* channels, int my_rank, int nranks, int num_elements) {
  int tid = threadIdx.x;
  int peer = 1 - my_rank;  // For 2 ranks: rank 0 talks to rank 1, and vice versa

  // Each rank owns one half of the buffer. Compute the byte offset and size.
  uint64_t size_per_rank = (uint64_t)num_elements * sizeof(int) / nranks;
  uint64_t my_offset = size_per_rank * my_rank;

  // put(): Copy our portion of the buffer to the remote GPU's buffer at the same offset.
  // Multiple threads cooperate on the copy for better bandwidth.
  channels[peer].put(my_offset, my_offset, size_per_rank, tid, blockDim.x);

  // Make sure all threads have finished the put() before we signal.
  __syncthreads();

  // signal() + wait(): Thread 0 tells the peer "data is ready" and waits for the peer's signal.
  if (tid == 0) {
    channels[peer].signal();
    channels[peer].wait();
  }
}
