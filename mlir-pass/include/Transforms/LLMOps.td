//===- LLMOps.td - LLM dialect operation definitions --------*- tablegen -*-===//
//
// Declares the custom `llm` dialect and the `llm.fused_rmsnorm_linear` op.
// TableGen generates the C++ boilerplate from this file.
//
//===----------------------------------------------------------------------===//

#ifndef LLM_OPS_TD
#define LLM_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def LLM_Dialect : Dialect {
  let name = "llm";
  let summary = "LLM kernel fusion dialect";
  let description = [{
    Custom dialect that hosts fused operations for large-language-model
    inference workloads.  Operations in this dialect lower to external CUDA
    kernel calls registered via the LLVM runtime.
  }];
  let cppNamespace = "::mlir::llm";
}

//===----------------------------------------------------------------------===//
// Base class
//===----------------------------------------------------------------------===//

class LLM_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLM_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// llm.fused_rmsnorm_linear
//===----------------------------------------------------------------------===//

def LLM_FusedRMSNormLinearOp : LLM_Op<"fused_rmsnorm_linear",
    [NoMemoryEffect]> {

  let summary = "Fused RMSNorm + linear projection — single HBM pass";

  let description = [{
    Computes:
      y_norm = RMSNorm(x, w_norm, epsilon)   // in-place over D_in elements
      y      = y_norm @ w_proj.T             // (B*S, D_in) x (D_out, D_in).T

    Inputs
    ------
    x       : f16 tensor of shape [B*S, D_in]   — input activations
    w_norm  : f16 tensor of shape [D_in]         — RMSNorm scale
    w_proj  : f16 tensor of shape [D_out, D_in]  — linear weight
    epsilon : f32 scalar attribute               — RMSNorm stability constant

    Output
    ------
    result  : f16 tensor of shape [B*S, D_out]

    Verifier checks:
    - x.dim(1) == w_norm.dim(0)   (inner dimension matches)
    - x.dim(1) == w_proj.dim(1)   (projection reads full hidden dim)
  }];

  let arguments = (ins
    AnyRankedTensor : $x,
    AnyRankedTensor : $w_norm,
    AnyRankedTensor : $w_proj,
    F32Attr         : $epsilon
  );

  let results = (outs AnyRankedTensor : $result);

  let assemblyFormat = [{
    `(` $x `,` $w_norm `,` $w_proj `,` $epsilon `)` attr-dict
    `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

#endif // LLM_OPS_TD
