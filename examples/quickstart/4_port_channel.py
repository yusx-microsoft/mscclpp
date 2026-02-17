# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# =============================================================================
# Example 4: Port Channel — proxy-based GPU data transfer
# =============================================================================
# Demonstrates GPU-to-GPU data transfer using PortChannel, which uses a
# host-side proxy thread to handle communication. Unlike MemoryChannel,
# PortChannel works over any transport (NVLink, InfiniBand, Ethernet) and
# only needs a single GPU thread for communication.
#
# Each rank fills its portion of a buffer, then uses a CUDA kernel with
# PortChannel's putWithSignalAndFlush() + wait() to exchange data.
#
# Run: mpirun -np 2 python 4_port_channel.py
# Requires: mscclpp, mpi4py, cupy
# =============================================================================

import os

import cupy as cp
from mpi4py import MPI
from mscclpp import CommGroup, GpuBuffer, ProxyService, Transport
from mscclpp.utils import KernelBuilder, pack

NELEM = 1024  # Total number of int32 elements in the buffer


def main():
    # Step 1: Initialize MPI and set the GPU for this rank
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank
    nranks = mpi_comm.size
    assert nranks == 2, "This example requires exactly 2 ranks (mpirun -np 2)"
    cp.cuda.Device(rank).use()

    # Step 2: Create a CommGroup and connections using CudaIpc transport
    group = CommGroup(mpi_comm=mpi_comm)
    all_ranks = list(range(nranks))
    connections = group.make_connection(all_ranks, Transport.CudaIpc)
    print(f"[Rank {rank}] Connections established")

    # Step 3: Allocate a GPU buffer and fill our portion with our rank + 1
    memory = GpuBuffer(NELEM, dtype=cp.int32)
    nelem_per_rank = NELEM // nranks
    memory[nelem_per_rank * rank : nelem_per_rank * (rank + 1)] = rank + 1

    # Step 4: Create a ProxyService — this starts a host-side thread that
    # polls for GPU requests and performs the actual data transfers
    proxy_service = ProxyService()

    # Step 5: Create PortChannels — the proxy service manages the semaphores
    # and memory registrations needed for communication
    channels = group.make_port_channels(proxy_service, memory, connections)

    # Step 6: Compile the CUDA kernel at runtime
    file_dir = os.path.dirname(os.path.abspath(__file__))
    kernel = KernelBuilder(
        file="4_port_channel_kernel.cu",
        kernel_name="port_channel_put",
        file_dir=file_dir,
    ).get_compiled_kernel()

    # Step 7: Pack the kernel arguments (same pattern as Memory Channel)
    first_channel = next(iter(channels.values()))
    handle_size = len(first_channel.device_handle().raw)
    device_handles = []
    for r in range(nranks):
        if r == rank:
            device_handles.append(bytes(handle_size))  # placeholder for self
        else:
            device_handles.append(channels[r].device_handle().raw)
    d_channels = cp.asarray(memoryview(b"".join(device_handles)), dtype=cp.uint8)

    params = pack(d_channels, rank, nranks, NELEM)

    # Step 8: Start the proxy, synchronize, launch the kernel, then stop the proxy
    proxy_service.start_proxy()
    group.barrier()
    kernel.launch_kernel(params, nblocks=1, nthreads=1024, shared=0, stream=None)
    cp.cuda.runtime.deviceSynchronize()
    proxy_service.stop_proxy()
    group.barrier()

    # Step 9: Verify — after the exchange, both halves of the buffer should be filled
    expected = cp.zeros(NELEM, dtype=cp.int32)
    for r in range(nranks):
        expected[nelem_per_rank * r : nelem_per_rank * (r + 1)] = r + 1

    assert cp.array_equal(memory, expected), f"[Rank {rank}] Data mismatch!"
    print(f"[Rank {rank}] Port channel transfer succeeded! Buffer: {memory}")


if __name__ == "__main__":
    main()
