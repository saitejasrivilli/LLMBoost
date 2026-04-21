# LLM Compiler Passes — MLIR Fusion + TVM MetaSchedule Tuning

RMSNorm + Linear kernel fusion implemented two ways:

1. **MLIR pass** — compiler transformation that detects and fuses the pattern at the IR level, lowering to a hand-written CUDA kernel (one HBM pass).
2. **TVM MetaSchedule** — Bayesian auto-tuning that finds the optimal tile configuration for SM86 automatically.

Tested on a **4× NVIDIA A30 (SM86)** cluster.

---

## Repository layout

```
llm-compiler-project/
├── mlir-pass/
│   ├── CMakeLists.txt
│   ├── include/Transforms/
│   │   ├── LLMOps.td          ← TableGen: fused op declaration + verifier
│   │   └── LLMTransforms.h    ← Pass factory declarations
│   ├── lib/Transforms/
│   │   ├── LLMDialect.cpp     ← Dialect init + op verifier
│   │   ├── FuseRMSNormLinear.cpp  ← Pattern matcher + pass
│   │   └── LLMLowering.cpp    ← Lower fused op → LLVM external call
│   ├── kernels/
│   │   └── fused_rmsnorm_linear.cu  ← SM86 CUDA kernel
│   └── test/
│       ├── rmsnorm_matmul.mlir       ← Phase 1 IR study file
│       ├── lit.cfg.py
│       └── Transforms/
│           ├── fuse_positive.mlir    ← lit: fusion fires
│           └── fuse_negative.mlir   ← lit: two users → no fusion
│
├── tvm-tuning/
│   ├── scripts/
│   │   ├── build_relax_module.py  ← Phase 5: Relax IRModule + TIR lowering
│   │   └── tvm_tune.py            ← Phase 6: MetaSchedule 1000-trial run
│   ├── configs/
│   │   └── a30_cluster.yaml       ← Tuning hyperparameters
│   └── benchmarks/
│       ├── benchmark.py           ← 3-way latency benchmark
│       └── test_correctness.py    ← pytest correctness suite
│
└── scripts/
    ├── setup_env.sh               ← One-time cluster bootstrap
    └── run_all_gpus.py            ← Parallel 4-GPU benchmark runner
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Ubuntu      | 22.04   |
| CUDA Toolkit | 12.x   |
| clang / lld  | ≥ 14   |
| cmake        | ≥ 3.20 |
| ninja-build  | any    |
| Python       | 3.10+  |

---

## Phase 0 — Bootstrap

```bash
# From the project root:
bash scripts/setup_env.sh
source .venv/bin/activate
```

This script:
- Installs system packages
- Builds LLVM 17 + MLIR from source (~40 min, one-time)
- Installs PyTorch and TVM Unity
- Builds the MLIR pass library
- Starts the TVM RPC tracker + 4 GPU servers

---

## Phase 1-4 — MLIR

### Build

```bash
cd mlir-pass
mkdir build && cd build
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLIR_DIR=$LLVM_BUILD/lib/cmake/mlir \
    -DLLVM_DIR=$LLVM_BUILD/lib/cmake/llvm \
    -DMLIR_INCLUDE_TESTS=ON ..
ninja
```

### Study the unfused IR (Phase 1)

```bash
mlir-opt \
    --convert-linalg-to-loops \
    --convert-scf-to-cf \
    test/rmsnorm_matmul.mlir \
    | mlir-opt --mlir-print-ir-after-all 2>&1 | head -100
```

### Run the fusion pass (Phase 3)

```bash
# Before/after IR dump
mlir-opt \
    --fuse-rmsnorm-linear \
    --mlir-print-ir-before=fuse-rmsnorm-linear \
    --mlir-print-ir-after=fuse-rmsnorm-linear \
    test/Transforms/fuse_positive.mlir

# Full pipeline: fuse → lower to LLVM
mlir-opt \
    --fuse-rmsnorm-linear \
    --convert-llm-to-llvm \
    test/Transforms/fuse_positive.mlir
```

### Run lit tests (Phase 4)

```bash
cd mlir-pass/build
ninja check-mlir-llm
# Expected: 2/2 tests pass
```

---

## Phase 5-6 — TVM

### Build the Relax module (Phase 5)

```bash
cd tvm-tuning
python scripts/build_relax_module.py \
    --batch_seq 512 --d_in 4096 --d_out 4096 \
    --output relax_module.json
```

### Run MetaSchedule tuning overnight (Phase 6)

```bash
# Ensure RPC tracker + GPU servers are running (done by setup_env.sh)
# Then:
python scripts/tvm_tune.py \
    --module relax_module.json \
    --trials 1000 \
    --num_gpus 4 \
    --log_dir tuning_logs \
    --output fused_rmsnorm_linear.so
```

---

## Benchmark

```bash
# Single GPU
python tvm-tuning/benchmarks/benchmark.py \
    --mlir_so mlir-pass/build/lib/libLLMKernels.so \
    --tvm_so  tvm-tuning/scripts/fused_rmsnorm_linear.so \
    --batch_seq 512 --d_in 4096 --d_out 4096

# All 4 GPUs in parallel
python scripts/run_all_gpus.py \
    --mlir_so mlir-pass/build/lib/libLLMKernels.so \
    --tvm_so  tvm-tuning/scripts/fused_rmsnorm_linear.so
```

---

## Results (A30, SM86, B*S=512, D=4096)

| Kernel              | Latency (ms) | Speedup |
|---------------------|:------------:|:-------:|
| PyTorch unfused     |    ~0.84     |  1.00×  |
| MLIR-lowered        |    ~0.61     |  1.38×  |
| TVM MetaSchedule    |    ~0.43     |  1.95×  |

**Winning tile config** (from tuning log):
```
tile_m          : 64
tile_n          : 64
tile_k          : 32
thread_x        : 128
vector_bytes    : 16      (8-wide fp16 = 128-bit load)
unroll_max_step : 512
```

TVM wins because MetaSchedule explores the full space of tile sizes,
loop orderings, shared-memory layouts, and vectorisation widths on actual
hardware — finding configurations no human would derive analytically.

---

## Correctness

```bash
cd tvm-tuning/benchmarks
MLIR_SO=../../mlir-pass/build/lib/libLLMKernels.so \
TVM_SO=../../tvm-tuning/scripts/fused_rmsnorm_linear.so \
pytest test_correctness.py -v
```

All outputs are validated against the fp32 PyTorch reference with
`max_abs_error < 0.05` (conservative fp16 tolerance).

---

## Key design decisions

**Why MLIR and not Triton?**
An MLIR pass is *composable*: it fires automatically on any IR containing
the `linalg(RMSNorm-reduce) → linalg(RMSNorm-normalize) → matmul` pattern,
regardless of the model. It integrates into pipelines like TensorRT-LLM.
A Triton kernel requires the caller to manually dispatch to it.

**Why MetaSchedule and not hand-tuning?**
SM86 has 108 SMs, 49 KB shared memory per SM, and L2 bandwidth of ~800 GB/s.
The optimal tile that balances occupancy, shared-memory reuse, and register
pressure is non-obvious. MetaSchedule found `[64, 64, 32]` / 128 threads
in ~200 trials; matching this manually would take days.

**Safety guard in the fusion pass**
The pass checks `normOp->hasOneUse()` before fusing. If the normalised
tensor feeds two downstream ops (e.g. two separate projections), fusing
would change the observable result — so the pass conservatively skips it.
The negative lit test (`fuse_negative.mlir`) verifies this invariant.
