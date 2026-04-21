set(LLVM_TARGET_DEFINITIONS LLMOps.td)
mlir_tablegen(LLMOps.h.inc -gen-op-decls)
mlir_tablegen(LLMOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRLLMOpsIncGen)
