# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# =============================================================================
# Example 2: GPU Ping-Pong (CPU-side data exchange)
# =============================================================================
# Demonstrates MSCCL++ bootstrap and communicator setup using MPI. Two ranks
# exchange a small numpy array via CPU-side send()/recv() to verify connectivity.
#
# Run: mpirun -np 2 python 2_gpu_ping_pong.py
# Requires: mscclpp, mpi4py, numpy
# =============================================================================

import numpy as np
from mpi4py import MPI
from mscclpp import CommGroup


def main():
    # Step 1: Initialize MPI — each process gets a unique rank
    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.rank
    world_size = mpi_comm.size
    assert world_size == 2, "This example requires exactly 2 ranks (mpirun -np 2)"

    # Step 2: Create a CommGroup — MSCCL++'s high-level wrapper that sets up
    # the TCP bootstrap and communicator using MPI for initial coordination
    group = CommGroup(mpi_comm=mpi_comm)
    print(f"[Rank {group.my_rank}] CommGroup created (world_size={group.nranks})")

    # Step 3: Prepare data to send — each rank sends its rank number
    send_buf = np.array([rank * 10 + 1, rank * 10 + 2, rank * 10 + 3], dtype=np.int32)
    recv_buf = np.zeros(3, dtype=np.int32)
    peer = 1 - rank  # rank 0 talks to rank 1, and vice versa
    tag = 0

    # Step 4: Exchange data via CPU-side bootstrap send/recv
    # Note: send() and recv() must be called in matching order to avoid deadlock
    if rank == 0:
        group.send(send_buf, peer=peer, tag=tag)
        group.recv(recv_buf, peer=peer, tag=tag)
    else:
        group.recv(recv_buf, peer=peer, tag=tag)
        group.send(send_buf, peer=peer, tag=tag)

    print(f"[Rank {rank}] Sent {send_buf}, Received {recv_buf}")

    # Step 5: Verify the data
    expected = np.array([peer * 10 + 1, peer * 10 + 2, peer * 10 + 3], dtype=np.int32)
    assert np.array_equal(recv_buf, expected), f"Data mismatch! Expected {expected}, got {recv_buf}"

    # Step 6: Synchronize all ranks before exiting
    group.barrier()
    print(f"[Rank {rank}] Ping-pong succeeded!")


if __name__ == "__main__":
    main()
