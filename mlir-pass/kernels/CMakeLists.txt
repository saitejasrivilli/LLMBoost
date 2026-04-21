add_library(LLMKernels SHARED fused_rmsnorm_linear.cu)
target_compile_options(LLMKernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xptxas -v>
)
set_target_properties(LLMKernels PROPERTIES
    CUDA_ARCHITECTURES "80;86"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
target_include_directories(LLMKernels PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(LLMKernels PRIVATE CUDA::cublas)
