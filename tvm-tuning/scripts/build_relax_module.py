"""
tvm_tuning/scripts/build_relax_module.py
=========================================
Phase 5: Define the fused RMSNorm + linear computation as a TVM Relax IRModule,
lower it to TIR, and print the resulting PrimFunc for inspection before tuning.

Usage:
  python build_relax_module.py \
      --batch_seq 512 --d_in 4096 --d_out 4096 \
      --epsilon 1e-5 --dtype float16

The script does NOT start tuning — that is tvm_tune.py.
"""

import argparse
import sys
import tvm
from tvm import relax, te, tir
from tvm.relax.frontend import nn
import tvm.script
from tvm.script import relax as R, tir as T


def build_relax_module(batch_seq: int, d_in: int, d_out: int,
                       epsilon: float, dtype: str) -> tvm.IRModule:
    """
    Build and return a TVM IRModule containing one Relax function:
        fused_rmsnorm_linear(x, w_norm, w_proj) -> y

    x      : [batch_seq, d_in]   dtype
    w_norm : [d_in]              dtype
    w_proj : [d_out, d_in]       dtype
    y      : [batch_seq, d_out]  dtype

    Numerics: upcast to fp32 for sum-of-squares / rsqrt, cast back to dtype.
    """
    bb = relax.BlockBuilder()

    # ── symbolic vars (static shapes for the A30 benchmark) ───────────────
    BS  = tvm.tir.IntImm("int64", batch_seq)
    DI  = tvm.tir.IntImm("int64", d_in)
    DO  = tvm.tir.IntImm("int64", d_out)

    x_ty      = relax.TensorStructInfo((BS, DI), dtype)
    wnorm_ty  = relax.TensorStructInfo((DI,),    dtype)
    wproj_ty  = relax.TensorStructInfo((DO, DI), dtype)

    with bb.function("fused_rmsnorm_linear",
                     params=["x", "w_norm", "w_proj"]):
        x      = relax.Var("x",      x_ty)
        w_norm = relax.Var("w_norm", wnorm_ty)
        w_proj = relax.Var("w_proj", wproj_ty)

        with bb.dataflow():
            # ── cast to fp32 for numerics ─────────────────────────────────
            x_fp32 = bb.emit(relax.op.astype(x, "float32"))
            wn_fp32 = bb.emit(relax.op.astype(w_norm, "float32"))

            # ── sum-of-squares per row:  [BS]  ────────────────────────────
            sq       = bb.emit(relax.op.multiply(x_fp32, x_fp32))
            sum_sq   = bb.emit(relax.op.sum(sq, axis=[1], keepdims=False))

            # ── rms scale:  [BS, 1]  ──────────────────────────────────────
            n_cols_v = relax.const(float(d_in), "float32")
            eps_v    = relax.const(epsilon, "float32")
            var      = bb.emit(relax.op.divide(sum_sq, n_cols_v))
            var_eps  = bb.emit(relax.op.add(var, eps_v))
            rms      = bb.emit(relax.op.rsqrt(var_eps))
            rms_exp  = bb.emit(relax.op.expand_dims(rms, axis=1))

            # ── normalize + scale ─────────────────────────────────────────
            normed   = bb.emit(relax.op.multiply(x_fp32, rms_exp))
            scaled   = bb.emit(relax.op.multiply(normed, wn_fp32))

            # ── cast back to target dtype ─────────────────────────────────
            scaled_t  = bb.emit(relax.op.astype(scaled, dtype))

            # ── linear projection:  [BS, D_out] ──────────────────────────
            # scaled_t: [BS, DI],  w_proj.T: [DI, DO]
            y = bb.emit(relax.op.matmul(scaled_t, relax.op.permute_dims(w_proj)))

            output = bb.emit_output(y)

        bb.emit_func_output(output)

    mod = bb.get()
    return mod


def lower_to_tir(mod: tvm.IRModule) -> tvm.IRModule:
    """
    Lower the Relax module to TIR PrimFuncs.
    After this step mod["fused_rmsnorm_linear"] is a TIR PrimFunc with
    explicit loop nests — the input to MetaSchedule.
    """
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.ToNonDataflow()(mod)
    mod = relax.transform.CallTIRRewrite()(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_seq", type=int, default=512)
    ap.add_argument("--d_in",      type=int, default=4096)
    ap.add_argument("--d_out",     type=int, default=4096)
    ap.add_argument("--epsilon",   type=float, default=1e-5)
    ap.add_argument("--dtype",     type=str, default="float16")
    ap.add_argument("--output",    type=str, default="relax_module.json",
                    help="Path to save the serialised Relax IRModule")
    args = ap.parse_args()

    print(f"[build_relax_module] Building Relax IRModule")
    print(f"  batch_seq={args.batch_seq}  d_in={args.d_in}  "
          f"d_out={args.d_out}  dtype={args.dtype}  eps={args.epsilon}")

    mod = build_relax_module(
        args.batch_seq, args.d_in, args.d_out, args.epsilon, args.dtype)

    print("\n── Relax IR (before lowering) ──────────────────────────────────")
    print(mod.script())

    mod_lowered = lower_to_tir(mod)
    print("\n── TIR PrimFunc (after lowering) ───────────────────────────────")
    # Print only the fused function to keep output manageable
    for name, func in mod_lowered.functions.items():
        if isinstance(func, tir.PrimFunc):
            print(f"\n// {name}")
            print(mod_lowered[name].script())

    # Save for the tuning script
    tvm.save_json(mod_lowered, args.output)
    print(f"\n[build_relax_module] Saved lowered module → {args.output}")


if __name__ == "__main__":
    main()
