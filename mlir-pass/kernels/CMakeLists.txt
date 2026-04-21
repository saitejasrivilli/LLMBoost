set(CUTLASS_DIR /storage/gxg8313/Strassens/nova_env2/lib/python3.11/site-packages/cutlass_library/source)

add_library(LLMKernels SHARED
    fused_rmsnorm_linear.cu
    fused_rmsnorm_linear_bf16.cu
)
target_compile_options(LLMKernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas -v --expt-relaxed-constexpr>
)
set_target_properties(LLMKernels PROPERTIES
    CUDA_ARCHITECTURES "80;86"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_include_directories(LLMKernels PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
)
target_link_libraries(LLMKernels PRIVATE CUDA::cublas)
