# needle

## Introduction

Ground-up version of [needle](https://dlsyscourse.org) for learning purposes.

Learning goals are:

* [Phase 1] Understand how DL frameworks with dynamic graphs can be constructed.
* [Phase 1] Practice writing kernels for different HW architectures.
* [Phase 1] Practice optimizing kernels for performance.
* [Phase 2] Understand how frontend frameworks are interfaced with DL compilers.
* [Phase 2] Understand what graph optimization techniques are used in practice.
* [Phase 2] Understand the code flow from StableHLO, to MLIR and to the actual kernels.

## Current plan

### Phase 1

- [ ] Implement n-dimensional array interface (for CPU, GPU and TPU).
- [ ] Allow PyTorch-like dynamic graph construction with the Tensor interface.
- [ ] Support auto-differentiation for the building blocks.
- [ ] Support building larger modules with the building blocks, such that the
      modules can also be auto-differentiated.
- [ ] Support basic datasets and dataloaders.
- [ ] Support basic optimizers.
- [ ] For proof-of-concept, implement ResNet-50 and train on mini-ImageNet.

### Phase 2

- [ ] Support JIT compilation (using StableHLO).
- [ ] TODO
