#include "Transforms/LLMTransforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fuse-rmsnorm-linear"

using namespace mlir;
using namespace mlir::linalg;

namespace {

static bool isRMSNormReduceBody(linalg::GenericOp op) {
  auto iterTypes = op.getIteratorTypesArray();
  if (iterTypes.size() < 2) return false;
  if (iterTypes.back() != utils::IteratorType::reduction) return false;
  for (size_t i = 0; i + 1 < iterTypes.size(); ++i)
    if (iterTypes[i] != utils::IteratorType::parallel) return false;
  Block &body = op.getRegion().front();
  SmallVector<Operation *> bodyOps;
  for (auto &op : body.without_terminator()) bodyOps.push_back(&op);
  if (bodyOps.size() != 2) return false;
  if (!isa<arith::MulFOp>(bodyOps[0])) return false;
  if (!isa<arith::AddFOp>(bodyOps[1])) return false;
  return true;
}

static bool isRMSNormNormalizeBody(linalg::GenericOp op) {
  auto iterTypes = op.getIteratorTypesArray();
  if (iterTypes.size() < 2) return false;
  for (auto &it : iterTypes)
    if (it != utils::IteratorType::parallel) return false;
  Block &body = op.getRegion().front();
  bool hasRsqrt = false;
  int mulCount = 0;
  for (auto &bop : body.without_terminator()) {
    if (isa<math::RsqrtOp>(&bop) || isa<math::SqrtOp>(&bop)) hasRsqrt = true;
    if (isa<arith::MulFOp>(&bop)) mulCount++;
  }
  return hasRsqrt && mulCount >= 2;
}

struct FuseRMSNormLinearPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    auto normOp = matmul.getInputs()[0].getDefiningOp<linalg::GenericOp>();
    if (!normOp || !isRMSNormNormalizeBody(normOp)) return failure();
    if (!normOp->hasOneUse()) return failure();
    auto reduceOp = normOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
    if (!reduceOp || !isRMSNormReduceBody(reduceOp)) return failure();

    Value x     = reduceOp.getInputs()[0];
    Value wNorm = normOp.getInputs()[1];
    Value wProj = matmul.getInputs()[1];

    FloatAttr epsilonAttr;
    normOp.getRegion().front().walk([&](arith::ConstantOp cst) {
      if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
        epsilonAttr = fa;
    });
    if (!epsilonAttr)
      epsilonAttr = rewriter.getF32FloatAttr(1e-5f);

    auto resultTy = matmul.getOutputs()[0].getType();
    rewriter.setInsertionPoint(matmul);
    auto fused = rewriter.create<mlir::llm::LLM_FusedRMSNormLinearOp>(
        matmul.getLoc(), resultTy, x, wNorm, wProj, epsilonAttr);
    rewriter.replaceOp(matmul, fused.getResult());
    if (normOp->use_empty())   rewriter.eraseOp(normOp);
    if (reduceOp->use_empty()) rewriter.eraseOp(reduceOp);
    return success();
  }
};

struct FuseRMSNormLinearPass
    : public PassWrapper<FuseRMSNormLinearPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseRMSNormLinearPass)
  StringRef getArgument()    const override { return "fuse-rmsnorm-linear"; }
  StringRef getDescription() const override {
    return "Fuse linalg RMSNorm + matmul into llm.fused_rmsnorm_linear";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseRMSNormLinearPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::llm::createFuseRMSNormLinearPass() {
  return std::make_unique<FuseRMSNormLinearPass>();
}

void mlir::llm::registerLLMPasses() {
  PassRegistration<FuseRMSNormLinearPass>();
}
