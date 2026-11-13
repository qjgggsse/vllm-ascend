# SPDX-License-Identifier: Apache-2.0
# Share the Torch ABI; compile only the kernel for the selected architecture.
if(SOC_VERSION MATCHES "ascend910_93.*")
    ascendc_library(vllm_ascend_hisparse STATIC op_kernel/arch22/hisparse_resolver.cpp)
elseif(SOC_VERSION MATCHES "ascend950")
    find_package(ASC REQUIRED CONFIG PATHS
        ${ASCEND_HOME_PATH}/lib64/cmake
        ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64/cmake
        NO_DEFAULT_PATH)
    set(CMAKE_ASC_ARCHITECTURES dav-3510)
    enable_language(ASC)
    ascendc_library(vllm_ascend_hisparse STATIC op_kernel/arch35/hisparse_resolver.cpp)
    target_include_directories(vllm_ascend_hisparse PRIVATE ${HISPARSE_SIMT_INCLUDE_DIR})
    target_compile_options(vllm_ascend_hisparse PRIVATE -O2)
    target_compile_definitions(vllm_ascend_hisparse PRIVATE HISPARSE_RESOLVER_UB_BYTES=175616)
endif()
