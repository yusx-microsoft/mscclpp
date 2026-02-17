# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# =============================================================================
# Example 3: Memory Channel — GPU-to-GPU data transfer
# =============================================================================
# Demonstrates direct GPU-to-GPU data transfer using MemoryChannel. This is
# the fastest channel type for intra-node communication over NVLink/xGMI.
#
# Each rank fills its portion of a buffer, then uses a CUDA kernel with
# MemoryChannel's put() + signal()/wait() to exchange data with the peer.
#
# Run: mpirun -np 2 python 3_memory_channel.py
# Requires: mscclpp, mpi4py, cupy
# =============================================================================

import os

import cupy as cp
from mpi4py import MPI
from mscclpp import CommGroup, GpuBuffer, Transport
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

    # Step 4: Create MemoryChannels — this registers GPU memory and exchanges
    # handles between ranks so each GPU can directly access the other's memory
    channels = group.make_memory_channels(memory, connections)

    # Step 5: Compile and launch the CUDA kernel
    # KernelBuilder compiles the .cu file at runtime using nvcc/hipcc
    file_dir = os.path.dirname(os.path.abspath(__file__))
    kernel = KernelBuilder(
        file="3_memory_channel_kernel.cu",
        kernel_name="memory_channel_put",
        file_dir=file_dir,
    ).get_compiled_kernel()

    # Step 6: Pack the kernel arguments
    # The channel device handles are packed into a contiguous GPU array so
    # the kernel can index them by rank.
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

    # Step 7: Synchronize all ranks, then launch the kernel
    group.barrier()
    kernel.launch_kernel(params, nblocks=1, nthreads=1024, shared=0, stream=None)
    cp.cuda.runtime.deviceSynchronize()
    group.barrier()

    # Step 8: Verify — after the exchange, both halves of the buffer should be filled
    expected = cp.zeros(NELEM, dtype=cp.int32)
    for r in range(nranks):
        expected[nelem_per_rank * r : nelem_per_rank * (r + 1)] = r + 1

    assert cp.array_equal(memory, expected), f"[Rank {rank}] Data mismatch!"
    print(f"[Rank {rank}] Memory channel transfer succeeded! Buffer: {memory}")


if __name__ == "__main__":
    main()
