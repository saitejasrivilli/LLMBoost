"""
tvm_tuning/benchmarks/benchmark.py
=====================================
Phase 6 Step 3 / Phase 4 Step 3:
Three-way latency benchmark on the 4× A30 cluster.

  PyTorch unfused  →  MLIR-lowered kernel  →  TVM-tuned .so

All measurements use the same input shapes, same GPU (cuda:0 by default),
CUDA events for timing, and a warm-up period before recording.

Usage:
  python benchmark.py \
      --batch_seq 512 \
      --d_in 4096 \
      --d_out 4096 \
      --tvm_so  ../../tvm-tuning/scripts/fused_rmsnorm_linear.so \
      --mlir_so ../../mlir-pass/build/lib/libLLMKernels.so \
      --warmup 50 \
      --iters  200 \
      --gpu    0

Output is printed as a Markdown table and saved to results.json.
"""

import argparse
import ctypes
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Timing utility
# ─────────────────────────────────────────────────────────────────────────────

def cuda_benchmark(fn, warmup: int, iters: int) -> float:
    """
    Returns mean latency in milliseconds using CUDA events.
    All timing is done on the device; no CPU synchronisation in the hot loop.
    """
    # Warm up
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    for _ in range(iters):
        fn()
    end_evt.record()
    torch.cuda.synchronize()

    return start_evt.elapsed_time(end_evt) / iters   # ms per iteration


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch unfused reference  (two HBM passes)
# ─────────────────────────────────────────────────────────────────────────────

def pytorch_rmsnorm_linear(
    x: torch.Tensor,        # [BS, D_in]  fp16
    w_norm: torch.Tensor,   # [D_in]      fp16
    w_proj: torch.Tensor,   # [D_out, D_in] fp16
    epsilon: float,
) -> torch.Tensor:
    # upcast to fp32 for stable norm
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    normed = (xf * rms * w_norm.float()).half()
    return F.linear(normed, w_proj)     # [BS, D_out]


# ─────────────────────────────────────────────────────────────────────────────
# MLIR-lowered kernel wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MLIRKernel:
    """
    Thin Python wrapper around the compiled CUDA shared library produced by
    the MLIR lowering pipeline.

    The .so exports:
      void fused_rmsnorm_linear_cuda(
          const __half *x,
          const __half *w_norm,
          const __half *w_proj,
                __half *y,
          int batch_seq, int d_in, int d_out,
          float epsilon);
    """

    def __init__(self, so_path: str):
        self._lib = ctypes.CDLL(so_path)
        fn = self._lib.fused_rmsnorm_linear_cuda
        fn.restype  = None
        fn.argtypes = [
            ctypes.c_void_p,   # x
            ctypes.c_void_p,   # w_norm
            ctypes.c_void_p,   # w_proj
            ctypes.c_void_p,   # y  (output, pre-allocated)
            ctypes.c_int,      # batch_seq
            ctypes.c_int,      # d_in
            ctypes.c_int,      # d_out
            ctypes.c_float,    # epsilon
        ]
        self._fn = fn

    def __call__(
        self,
        x: torch.Tensor,
        w_norm: torch.Tensor,
        w_proj: torch.Tensor,
        y: torch.Tensor,
        epsilon: float,
    ) -> None:
        bs, d_in = x.shape
        d_out    = w_proj.shape[0]
        self._fn(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(w_norm.data_ptr()),
            ctypes.c_void_p(w_proj.data_ptr()),
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_int(bs),
            ctypes.c_int(d_in),
            ctypes.c_int(d_out),
            ctypes.c_float(epsilon),
        )


# ─────────────────────────────────────────────────────────────────────────────
# TVM-tuned kernel wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TVMKernel:
    """
    Wrapper around the TVM-compiled .so exported by tvm_tune.py.
    Uses tvm.runtime to load and invoke the module.
    """

    def __init__(self, so_path: str, device_id: int = 0):
        import tvm
        import tvm.runtime as rt
        self._mod    = tvm.runtime.load_module(so_path)
        self._dev    = tvm.cuda(device_id)
        self._fn     = self._mod["fused_rmsnorm_linear"]

    def __call__(
        self,
        x: torch.Tensor,
        w_norm: torch.Tensor,
        w_proj: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        import tvm
        from tvm.nd import array as tvm_array
        dev = self._dev
        x_nd     = tvm_array(x.cpu().numpy(),     dev)
        wn_nd    = tvm_array(w_norm.cpu().numpy(), dev)
        wp_nd    = tvm_array(w_proj.cpu().numpy(), dev)
        y_nd     = tvm_array(y.cpu().numpy(),      dev)
        self._fn(x_nd, wn_nd, wp_nd, y_nd)
        # copy result back to torch tensor
        y.copy_(torch.from_numpy(y_nd.numpy()).to(y.device))


# ─────────────────────────────────────────────────────────────────────────────
# Numerical correctness check
# ─────────────────────────────────────────────────────────────────────────────

def check_correctness(ref: torch.Tensor, cand: torch.Tensor,
                      label: str, atol: float = 1e-2) -> bool:
    max_err = (ref.float() - cand.float()).abs().max().item()
    ok = max_err < atol
    status = "PASS" if ok else "FAIL"
    print(f"  correctness [{label}]: {status}  (max_abs_err={max_err:.4e})")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_seq", type=int, default=512)
    ap.add_argument("--d_in",      type=int, default=4096)
    ap.add_argument("--d_out",     type=int, default=4096)
    ap.add_argument("--epsilon",   type=float, default=1e-5)
    ap.add_argument("--tvm_so",    type=str,   default=None,
                    help="Path to TVM-compiled .so (skip if not built yet)")
    ap.add_argument("--mlir_so",   type=str,   default=None,
                    help="Path to MLIR-compiled libLLMKernels.so")
    ap.add_argument("--warmup",    type=int, default=50)
    ap.add_argument("--iters",     type=int, default=200)
    ap.add_argument("--gpu",       type=int, default=0)
    ap.add_argument("--output",    type=str, default="results.json")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    BS   = args.batch_seq
    DI   = args.d_in
    DO   = args.d_out
    EPS  = args.epsilon

    print(f"\n{'='*60}")
    print(f" Benchmark: RMSNorm + Linear")
    print(f" Shape: [{BS}, {DI}] x [{DI}, {DO}]")
    print(f" Device: {torch.cuda.get_device_name(args.gpu)}")
    print(f"{'='*60}\n")

    # ── Allocate tensors ──────────────────────────────────────────────────
    torch.manual_seed(0)
    x      = torch.randn(BS, DI, dtype=torch.float16, device=device)
    w_norm = torch.randn(DI,     dtype=torch.float16, device=device)
    w_proj = torch.randn(DO, DI, dtype=torch.float16, device=device)
    y_ref  = torch.empty(BS, DO, dtype=torch.float16, device=device)
    y_out  = torch.empty(BS, DO, dtype=torch.float16, device=device)

    results = {
        "device":    torch.cuda.get_device_name(args.gpu),
        "batch_seq": BS, "d_in": DI, "d_out": DO,
        "warmup":    args.warmup, "iters": args.iters,
        "kernels":   {}
    }

    # ── 1. PyTorch unfused (baseline) ─────────────────────────────────────
    print("── 1. PyTorch unfused (two HBM passes) ───────────────────────")

    def pt_fn():
        return pytorch_rmsnorm_linear(x, w_norm, w_proj, EPS)

    y_ref = pt_fn()   # reference output
    pt_ms = cuda_benchmark(pt_fn, args.warmup, args.iters)
    print(f"  latency: {pt_ms:.4f} ms")
    results["kernels"]["pytorch_unfused"] = {"latency_ms": pt_ms}

    # ── 2. MLIR-lowered kernel ─────────────────────────────────────────────
    if args.mlir_so and Path(args.mlir_so).exists():
        print("\n── 2. MLIR-lowered kernel ────────────────────────────────────")
        try:
            mlir_kernel = MLIRKernel(args.mlir_so)

            def mlir_fn():
                mlir_kernel(x, w_norm, w_proj, y_out, EPS)

            mlir_fn()
            check_correctness(y_ref, y_out, "MLIR")
            mlir_ms = cuda_benchmark(mlir_fn, args.warmup, args.iters)
            speedup  = pt_ms / mlir_ms
            print(f"  latency: {mlir_ms:.4f} ms  ({speedup:.2f}x vs PyTorch)")
            results["kernels"]["mlir"] = {
                "latency_ms": mlir_ms,
                "speedup_vs_pytorch": speedup,
            }
        except Exception as e:
            print(f"  [SKIP] MLIR kernel failed to load: {e}")
    else:
        print("\n── 2. MLIR-lowered kernel  [SKIPPED — .so not found] ─────────")

    # ── 3. TVM-tuned kernel ───────────────────────────────────────────────
    if args.tvm_so and Path(args.tvm_so).exists():
        print("\n── 3. TVM-tuned kernel ───────────────────────────────────────")
        try:
            tvm_kernel = TVMKernel(args.tvm_so, device_id=args.gpu)

            def tvm_fn():
                tvm_kernel(x, w_norm, w_proj, y_out)

            tvm_fn()
            check_correctness(y_ref, y_out, "TVM")
            tvm_ms  = cuda_benchmark(tvm_fn, args.warmup, args.iters)
            speedup = pt_ms / tvm_ms
            print(f"  latency: {tvm_ms:.4f} ms  ({speedup:.2f}x vs PyTorch)")
            results["kernels"]["tvm"] = {
                "latency_ms": tvm_ms,
                "speedup_vs_pytorch": speedup,
            }
        except Exception as e:
            print(f"  [SKIP] TVM kernel failed to load: {e}")
    else:
        print("\n── 3. TVM-tuned kernel  [SKIPPED — .so not found] ────────────")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" Results table  (device: {torch.cuda.get_device_name(args.gpu)})")
    print(f"{'='*60}")
    print(f" {'Kernel':<22}  {'Latency (ms)':>14}  {'Speedup':>9}")
    print(f" {'-'*22}  {'-'*14}  {'-'*9}")
    for name, info in results["kernels"].items():
        lat = info["latency_ms"]
        spd = info.get("speedup_vs_pytorch", 1.0)
        print(f" {name:<22}  {lat:>14.4f}  {spd:>8.2f}x")
    print(f"{'='*60}\n")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[benchmark] Results saved → {args.output}")


if __name__ == "__main__":
    main()
