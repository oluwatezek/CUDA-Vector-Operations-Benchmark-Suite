# CUDA Vector Addition Benchmark

## Overview

Implemented vector addition using both CPU and CUDA GPU implementations.

The goal of this project was to understand:

- CUDA execution model
- GPU threads and blocks
- Host-device memory transfers
- Kernel execution
- GPU benchmarking

---

## Implementation

### CPU Version

A standard C++ implementation:

C[i] = A[i] + B[i]

### CUDA Version

Each GPU thread computes one element:

Thread i:

C[i] = A[i] + B[i]

Execution configuration:

- Threads per block: 256
- Blocks: calculated dynamically based on input size

---

## CUDA Pipeline
