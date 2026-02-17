// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// =============================================================================
// Example 1: Hello MSCCL++
// =============================================================================
// The simplest possible MSCCL++ program. This creates a Context, two Endpoints
// (one per GPU), and connects them. No data is transferred — this just verifies
// that MSCCL++ is installed correctly and two GPUs can establish a connection.
//
// Build:  make            (from this directory)
// Run:    ./hello_mscclpp (requires 2 GPUs with peer-to-peer access)
// =============================================================================

#include <iostream>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>

int main() {
  // Step 1: Check that we have at least 2 GPUs
  int deviceCount;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    std::cerr << "Error: At least two GPUs are required." << std::endl;
    return 1;
  }
  std::cout << "Found " << deviceCount << " GPUs." << std::endl;

  // Step 2: Check peer-to-peer access between GPU 0 and GPU 1
  int canAccessPeer;
  MSCCLPP_CUDATHROW(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));
  if (!canAccessPeer) {
    std::cerr << "Error: GPU 0 cannot access GPU 1 peer-to-peer.\n"
              << "Check `nvidia-smi topo -m` — GPUs need NV# or PIX connectivity." << std::endl;
    return 1;
  }
  std::cout << "GPU 0 and GPU 1 can access each other." << std::endl;

  // Step 3: Create a Context — the top-level MSCCL++ object for single-process use
  auto ctx = mscclpp::Context::create();
  std::cout << "Context created." << std::endl;

  // Step 4: Create Endpoints — one per GPU, using CudaIpc transport (intra-node)
  mscclpp::Endpoint ep0 = ctx->createEndpoint({mscclpp::Transport::CudaIpc, {mscclpp::DeviceType::GPU, 0}});
  mscclpp::Endpoint ep1 = ctx->createEndpoint({mscclpp::Transport::CudaIpc, {mscclpp::DeviceType::GPU, 1}});
  std::cout << "Endpoints created for GPU 0 and GPU 1." << std::endl;

  // Step 5: Connect the endpoints to establish a Connection
  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  mscclpp::Connection conn0 = ctx->connect(ep0, ep1);

  MSCCLPP_CUDATHROW(cudaSetDevice(1));
  mscclpp::Connection conn1 = ctx->connect(ep1, ep0);
  std::cout << "Connections established between GPU 0 and GPU 1." << std::endl;

  std::cout << "\nSuccess! MSCCL++ is working. Your GPUs are ready for communication." << std::endl;
  return 0;
}
