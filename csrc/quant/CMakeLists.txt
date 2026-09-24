# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(OPTEST_NAME optest_${PKG_NAME})

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if (DEFINED ASCEND_OP_NAME AND NOT "${ASCEND_OP_NAME}" STREQUAL "")
        if (NOT "${ASCEND_OP_NAME}" STREQUAL "all" AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL")
            if (NOT ${SUB_DIR} IN_LIST ASCEND_OP_NAME)
                continue()
            endif ()
        endif ()
    endif ()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
        # The ops-nn host CMake registers the opdef interface, while this
        # package generates ACLNN from the op_host_aclnn source list.
        if(BUILD_OPEN_PROJECT)
            target_sources(op_host_aclnn PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/op_host/${SUB_DIR}_def.cpp)
        endif()
        # ops-nn's unchanged host CMake does not register the operator in
        # the ops-transformer package lists; register it at the category edge.
        set(COMPILED_OPS ${COMPILED_OPS} ${SUB_DIR} CACHE STRING "Compiled Ops" FORCE)
        set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}
            CACHE STRING "Compiled Ops Dirs" FORCE)
    else()
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/op_host/CMakeLists.txt")
            add_subdirectory(${SUB_DIR}/op_host)
        endif()
    endif()
endforeach()
