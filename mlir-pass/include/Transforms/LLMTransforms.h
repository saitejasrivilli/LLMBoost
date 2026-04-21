#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

// Dialect declarations (generated from LLMOps.td)
#include "mlir/IR/DialectImplementation.h"
namespace mlir {
namespace llm {
class LLM_Dialect : public ::mlir::Dialect {
public:
  explicit LLM_Dialect(::mlir::MLIRContext *context);
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("llm");
  }
  void initialize();
};
} // namespace llm
} // namespace mlir

#define GET_OP_CLASSES
#include "Transforms/LLMOps.h.inc"

namespace mlir {
namespace llm {
// Convenience alias matching old name used in source files
using LLM_FusedRMSNormLinearOp = FusedRMSNormLinearOp;

std::unique_ptr<OperationPass<func::FuncOp>> createFuseRMSNormLinearPass();
std::unique_ptr<OperationPass<ModuleOp>> createLLMToLLVMLoweringPass();
void registerLLMPasses();
} // namespace llm
} // namespace mlir
