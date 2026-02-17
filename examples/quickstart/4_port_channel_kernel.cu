// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// =============================================================================
// Companion CUDA kernel for Example 4: Port Channel
// =============================================================================
// This kernel uses PortChannel to copy data between GPUs via the host-side
// proxy. Unlike MemoryChannel, PortChannel works over any transport (NVLink,
// InfiniBand, etc.) and only needs a single GPU thread for communication.
//
// The pattern is:
//   1. putWithSignalAndFlush() — send data + signal in one call (proxy handles it)
//   2. wait() — wait for the peer's data to arrive
//
// Compiled at runtime by KernelBuilder in 4_port_channel.py.
// =============================================================================

#include <mscclpp/port_channel_device.hpp>

extern "C" __global__ void __launch_bounds__(1024, 1)
    port_channel_put(mscclpp::PortChannelDeviceHandle* channels, int my_rank, int nranks, int num_elements) {
  int tid = threadIdx.x;
  int peer = 1 - my_rank;  // For 2 ranks: rank 0 talks to rank 1

  // Each rank owns one half of the buffer.
  uint64_t size_per_rank = (uint64_t)num_elements * sizeof(int) / nranks;
  uint64_t my_offset = size_per_rank * my_rank;

  // Only one thread is needed for PortChannel operations — the proxy handles
  // the actual data movement on the host side.
  if (tid == 0) {
    // putWithSignalAndFlush(): Send our data to the peer and signal completion.
    // This is an all-in-one call: put + signal + flush (ensure proxy processes it).
    channels[peer].putWithSignalAndFlush(my_offset, my_offset, size_per_rank);

    // wait(): Block until the peer's data has arrived.
    channels[peer].wait();
  }
}
