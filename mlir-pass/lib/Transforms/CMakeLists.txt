add_mlir_library(MLIRLLMTransforms
    LLMDialect.cpp
    FuseRMSNormLinear.cpp
    LLMLowering.cpp
    DEPENDS
    MLIRLLMOpsIncGen
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRTransforms
    MLIRRewrite
)
