add_op_to_compiled_list()

set(TURBO_QUANT_SPARSE_FLASH_ATTENTION_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for this operator")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
    turbo_quant_sparse_flash_attention_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME TurboQuantSparseFlashAttention
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -fpermissive
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE turbo_quant_sparse_flash_attention ACLNNTYPE aclnn)
endif()
