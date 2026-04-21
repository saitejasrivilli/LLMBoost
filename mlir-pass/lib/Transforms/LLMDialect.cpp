#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "Transforms/LLMTransforms.h"

using namespace mlir;
using namespace mlir::llm;

#define GET_OP_CLASSES
#include "Transforms/LLMOps.cpp.inc"

mlir::llm::LLM_Dialect::LLM_Dialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect("llm", context,
                      ::mlir::TypeID::get<LLM_Dialect>()) {
  initialize();
}

void LLM_Dialect::initialize() {
  addOperations<FusedRMSNormLinearOp>();
}

LogicalResult FusedRMSNormLinearOp::verify() {
  auto xTy    = cast<RankedTensorType>(getX().getType());
  auto normTy = cast<RankedTensorType>(getWNorm().getType());
  auto projTy = cast<RankedTensorType>(getWProj().getType());
  if (xTy.getRank() != 2)
    return emitOpError("x must be a 2-D tensor [batch_seq, d_in]");
  if (normTy.getRank() != 1)
    return emitOpError("w_norm must be a 1-D tensor [d_in]");
  if (projTy.getRank() != 2)
    return emitOpError("w_proj must be a 2-D tensor [d_out, d_in]");
  int64_t dIn = xTy.getDimSize(1);
  if (normTy.getDimSize(0) != dIn)
    return emitOpError("w_norm dim(0) must equal x dim(1)");
  if (projTy.getDimSize(1) != dIn)
    return emitOpError("w_proj dim(1) must equal x dim(1)");
  auto resTy = cast<RankedTensorType>(getResult().getType());
  if (resTy.getRank() != 2)
    return emitOpError("result must be a 2-D tensor [batch_seq, d_out]");
  if (resTy.getDimSize(0) != xTy.getDimSize(0))
    return emitOpError("result dim(0) must equal x dim(0)");
  if (resTy.getDimSize(1) != projTy.getDimSize(0))
    return emitOpError("result dim(1) must equal w_proj dim(0)");
  return success();
}
