# MSCCL++ Quickstart Examples

A progressive set of minimal examples to get you started with MSCCL++. Each example focuses on one concept, is heavily commented, and is as short as possible.

## Prerequisites

- **Hardware**: At least 2 NVIDIA GPUs with peer-to-peer access (check with `nvidia-smi topo -m`)
- **MSCCL++ installed**: Either built from source or `pip install -e .` from the repo root
- **For C++ examples**: CUDA toolkit (nvcc) and `libmscclpp` linked
- **For Python examples**: `mpi4py`, `cupy`, and `mscclpp` Python package

Install Python dependencies:
```bash
pip install mpi4py cupy-cuda12x   # adjust cupy package for your CUDA version
pip install -e .                   # install mscclpp from repo root
```

## Examples

### 1. Hello MSCCL++ (C++)

The absolute simplest MSCCL++ program. Creates a `Context`, two `Endpoint`s, and connects them — confirming MSCCL++ is installed and two GPUs can talk.

```bash
cd examples/quickstart
make
./hello_mscclpp
```

### 2. GPU Ping-Pong (Python)

Bootstrap and communicator setup using MPI. Demonstrates CPU-side `send()`/`recv()` between two ranks to verify connectivity.

```bash
mpirun -np 2 python examples/quickstart/2_gpu_ping_pong.py
```

### 3. Memory Channel (Python + CUDA kernel)

GPU-to-GPU data transfer using `MemoryChannel` — the direct memory-mapped approach (best for NVLink/xGMI). Shows the `put()` + `signal()`/`wait()` pattern.

```bash
mpirun -np 2 python examples/quickstart/3_memory_channel.py
```

### 4. Port Channel (Python + CUDA kernel)

GPU-to-GPU data transfer using `PortChannel` — the proxy-based approach. Works over any transport (NVLink, InfiniBand, etc.). Shows `putWithSignalAndFlush()` + `wait()`.

```bash
mpirun -np 2 python examples/quickstart/4_port_channel.py
```

## Concepts at a Glance

| Concept | Where |
|---|---|
| Context & Endpoints (single-process) | Example 1 |
| Bootstrap & CommGroup (multi-process) | Examples 2–4 |
| CPU-side send/recv | Example 2 |
| MemoryChannel (direct GPU access) | Example 3 |
| PortChannel (proxy-based) | Example 4 |
| KernelBuilder (runtime CUDA compilation) | Examples 3–4 |

## Troubleshooting

- **"At least two GPUs are required"**: You need ≥2 GPUs. Check with `nvidia-smi`.
- **"GPU 0 cannot access GPU 1"**: GPUs must support peer-to-peer. Check with `nvidia-smi topo -m`.
- **MPI errors**: Make sure `mpi4py` is installed and `mpirun` is on your PATH.
- **Import errors**: Make sure `mscclpp` is installed (`pip install -e .` from repo root).
- **MSCCLPP_HOME not set**: If using a custom install path, set `MSCCLPP_HOME` to point to your installation.
