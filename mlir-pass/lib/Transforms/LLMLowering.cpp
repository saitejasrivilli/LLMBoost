#include "Transforms/LLMTransforms.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {

static constexpr llvm::StringRef kKernelSymbol = "fused_rmsnorm_linear_cuda";

static LLVMFuncOp getOrInsertKernelDecl(ModuleOp module, OpBuilder &b) {
  if (auto fn = module.lookupSymbol<LLVMFuncOp>(kKernelSymbol)) return fn;
  auto *ctx   = module.getContext();
  auto voidTy = LLVMVoidType::get(ctx);
  auto ptrTy  = LLVMPointerType::get(ctx, 0);
  auto i32Ty  = IntegerType::get(ctx, 32);
  auto f32Ty  = Float32Type::get(ctx);
  auto fnTy   = LLVMFunctionType::get(
      voidTy, {ptrTy,ptrTy,ptrTy,ptrTy,i32Ty,i32Ty,i32Ty,f32Ty}, false);
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());
  auto decl = b.create<LLVMFuncOp>(module.getLoc(), kKernelSymbol, fnTy);
  decl.setLinkage(Linkage::External);
  return decl;
}

struct FusedRMSNormLinearLoweringPattern : public ConversionPattern {
  FusedRMSNormLinearLoweringPattern(LLVMTypeConverter &tc)
      : ConversionPattern(
            tc,
            llvm::StringRef(mlir::llm::FusedRMSNormLinearOp::getOperationName()),
            /*benefit=*/1,
            &tc.getContext()) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto fusedOp = cast<mlir::llm::FusedRMSNormLinearOp>(op);
    auto loc     = fusedOp.getLoc();
    auto module  = op->getParentOfType<ModuleOp>();
    OpBuilder b(rewriter);
    auto kernelDecl = getOrInsertKernelDecl(module, b);

    auto xTy     = cast<RankedTensorType>(fusedOp.getX().getType());
    auto projTy  = cast<RankedTensorType>(fusedOp.getWProj().getType());
    int64_t batchSeq = xTy.getDimSize(0);
    int64_t dIn      = xTy.getDimSize(1);
    int64_t dOut     = projTy.getDimSize(0);

    auto i32     = rewriter.getI32Type();
    Value c_bs   = rewriter.create<arith::ConstantIntOp>(loc, batchSeq, i32);
    Value c_din  = rewriter.create<arith::ConstantIntOp>(loc, dIn,      i32);
    Value c_dout = rewriter.create<arith::ConstantIntOp>(loc, dOut,     i32);
    Value c_eps  = rewriter.create<arith::ConstantFloatOp>(
        loc, fusedOp.getEpsilon(), rewriter.getF32Type());

    auto ptrTy = LLVMPointerType::get(rewriter.getContext(), 0);
    auto castPtr = [&](Value v) -> Value {
      return rewriter.create<UnrealizedConversionCastOp>(
          loc, TypeRange{ptrTy}, ValueRange{v}).getResult(0);
    };

    Value xPtr    = castPtr(operands[0]);
    Value normPtr = castPtr(operands[1]);
    Value projPtr = castPtr(operands[2]);

    auto resTy  = cast<RankedTensorType>(fusedOp.getResult().getType());
    auto f16Ty  = Float16Type::get(rewriter.getContext());
    auto arrTy  = LLVMArrayType::get(f16Ty, resTy.getNumElements());
    Value one   = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    Value outBuf = rewriter.create<AllocaOp>(loc, ptrTy, arrTy, one);

    SmallVector<Value> args;
    args.push_back(xPtr);    args.push_back(normPtr);
    args.push_back(projPtr); args.push_back(outBuf);
    args.push_back(c_bs);    args.push_back(c_din);
    args.push_back(c_dout);  args.push_back(c_eps);
    rewriter.create<CallOp>(loc, kernelDecl, ValueRange(args));

    Value result = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{resTy}, ValueRange{outBuf}).getResult(0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LLMToLLVMLoweringPass
    : public PassWrapper<LLMToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLMToLLVMLoweringPass)
  StringRef getArgument()    const override { return "convert-llm-to-llvm"; }
  StringRef getDescription() const override {
    return "Lower llm.fused_rmsnorm_linear to external CUDA kernel call";
  }
  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<mlir::llm::LLM_Dialect>();
    LLVMTypeConverter typeConv(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<FusedRMSNormLinearLoweringPattern>(typeConv);
    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::llm::createLLMToLLVMLoweringPass() {
  return std::make_unique<LLMToLLVMLoweringPass>();
}
