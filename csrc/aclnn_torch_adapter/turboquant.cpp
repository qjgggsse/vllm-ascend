/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file turboquant.cpp
 * \brief
 */

#include <torch/extension.h>
#include "op_api_common.h"

namespace op_api {

// npu tensor max size
const int SIZE = 8;
const int DIM_0 = 0;
const int DIM_1 = 1;
const int DIM_2 = 2;
const int DIM_3 = 3;
const int DIM_4 = 4;

std::tuple<at::Tensor, at::Tensor> ConstructMixedQuantSparseFlashMlaAttenOutTensor(
    const at::Tensor &q, const at::Tensor &oriKv, std::string layoutQStr, std::string layoutKvStr,
    const uint64_t &ropeHeadDim, bool returnSoftmaxLse, int64_t quantMode)
{
    TORCH_CHECK(layoutQStr == "BSND" || layoutQStr == "TND", "The layout of query only support BSND and TND, but got ",
                layoutQStr);
    for (auto i = 0; i < q.sizes().size(); i++) {
        const bool turboQuantEmptyQuery = quantMode == 3 && layoutQStr == "TND" && i == DIM_0 && q.size(i) == 0;
        TORCH_CHECK(q.size(i) > 0 || turboQuantEmptyQuery,
                    "All values within query's shape should be greater "
                    "than 0, but shape[",
                    i, "] is ", q.size(i));
    }
    at::SmallVector<int64_t, SIZE> attenOutSize;
    at::SmallVector<int64_t, SIZE> softmaxLseSize;
    if (layoutQStr == "BSND") {
        TORCH_CHECK(q.dim() == DIM_4, "When the layout of query is BSND, the query dimension must be 4, but got ",
                    q.dim());
        // atten_out_size = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2), q.size(DIM_3) - rope_head_dim};
        attenOutSize = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2), q.size(DIM_3)};
    } else {
        TORCH_CHECK(q.dim() == DIM_3, "When the layout of query is TND, the query dimension must be 3, but got ",
                    q.dim());
        // atten_out_size = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2) - rope_head_dim};
        attenOutSize = {q.size(DIM_0), q.size(DIM_1), q.size(DIM_2)};
    }
    at::Tensor attenOut = at::empty(attenOutSize, q.options().dtype(q.dtype()));

    if (returnSoftmaxLse) {
        TORCH_CHECK(oriKv.size(DIM_1) > 0, "oriKv.size(DIM_1) must be greater than 0, but got ", oriKv.size(DIM_1));
        TORCH_CHECK(oriKv.size(DIM_2) > 0, "oriKv.size(DIM_2) must be greater than 0, but got ", oriKv.size(DIM_2));
        if (layoutQStr == "BSND") {
            int64_t dim0 = static_cast<int64_t>(q.size(DIM_0));
            int64_t dim1 = static_cast<int64_t>(oriKv.size(DIM_2));
            int64_t dim2 = static_cast<int64_t>(q.size(DIM_1));
            int64_t dim3 = static_cast<int64_t>(q.size(DIM_2)) / static_cast<int64_t>(oriKv.size(DIM_2));
            softmaxLseSize = {dim0, dim1, dim2, dim3};
        } else {
            if (layoutKvStr == "PA_BBND") {
                int64_t dim0 = static_cast<int64_t>(oriKv.size(DIM_2));
                int64_t dim1 = static_cast<int64_t>(q.size(DIM_0));
                int64_t dim2 = static_cast<int64_t>(q.size(DIM_1)) / static_cast<int64_t>(oriKv.size(DIM_2));
                softmaxLseSize = {dim0, dim1, dim2};
            } else {
                int64_t dim0 = static_cast<int64_t>(oriKv.size(DIM_1));
                int64_t dim1 = static_cast<int64_t>(q.size(DIM_0));
                int64_t dim2 = static_cast<int64_t>(q.size(DIM_1)) / static_cast<int64_t>(oriKv.size(DIM_1));
                softmaxLseSize = {dim0, dim1, dim2};
            }
        }
    } else {
        // 不返回时tensor传空
        softmaxLseSize = {0};
    }
    at::Tensor softmaxLse = at::empty(softmaxLseSize, q.options().dtype(torch::kFloat32));

    return std::tuple<at::Tensor, at::Tensor>(attenOut, softmaxLse);
}

std::tuple<at::Tensor, at::Tensor> MixedQuantSparseFlashMla(
    const at::Tensor &q, const c10::optional<at::Tensor> &oriKv, const c10::optional<at::Tensor> &cmpKv,
    const c10::optional<at::Tensor> &oriSparseIndices, const c10::optional<at::Tensor> &cmpSparseIndices,
    const c10::optional<at::Tensor> &oriBlockTable, const c10::optional<at::Tensor> &cmpBlockTable,
    const c10::optional<at::Tensor> &cuSeqlensQ, const c10::optional<at::Tensor> &cuSeqlensOriKv,
    const c10::optional<at::Tensor> &cuSeqlensCmpKv, const c10::optional<at::Tensor> &sequsedQ,
    const c10::optional<at::Tensor> &sequsedOriKv, const c10::optional<at::Tensor> &sequsedCmpKv,
    const c10::optional<at::Tensor> &cmpResidualKv, const c10::optional<at::Tensor> &oriTopkLength,
    const c10::optional<at::Tensor> &cmpTopkLength, const c10::optional<at::Tensor> &sinks,
    const c10::optional<at::Tensor> &metadata, int64_t quantMode, int64_t ropeHeadDim, double softmaxScale,
    int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
    c10::string_view layoutQ, c10::string_view layoutKv, int64_t topkValueMode, bool returnSoftmaxLse)
{
    std::string layoutQStr = std::string(layoutQ);
    std::string layoutKvStr = std::string(layoutKv);
    const bool turboQuantEmptyQuery = quantMode == 3 && layoutQStr == "TND" && q.dim() == DIM_3 && q.size(DIM_0) == 0;
    TORCH_CHECK(q.numel() > 0 || turboQuantEmptyQuery, "Tensor query is empty.");
    TORCH_CHECK(oriKv.has_value(), "ori_kv must be provided.");
    const at::Tensor &oriKvVal = *oriKv;
    // convert str
    char *layoutQPtr = const_cast<char *>(layoutQStr.c_str());
    char *layoutKvPtr = const_cast<char *>(layoutKvStr.c_str());

    // construct the atten_out tensor
    std::tuple<at::Tensor, at::Tensor> mixedQuantSparseFlashMlaAttenOut =
        op_api::ConstructMixedQuantSparseFlashMlaAttenOutTensor(q, oriKvVal, layoutQStr, layoutKvStr, ropeHeadDim,
                                                                returnSoftmaxLse, quantMode);
    at::Tensor attenOut = std::get<0>(mixedQuantSparseFlashMlaAttenOut);
    at::Tensor softmaxLse = std::get<1>(mixedQuantSparseFlashMlaAttenOut);

    at::Tensor nullTensor;
    auto oriKvValue = oriKv.has_value() ? oriKv.value() : nullTensor;
    auto cmpKvValue = cmpKv.has_value() ? cmpKv.value() : nullTensor;
    if (q.is_meta()) return {attenOut, softmaxLse};
    EXEC_NPU_CMD(aclnnMixedQuantSparseFlashMla, q, oriKvValue, cmpKvValue, oriSparseIndices, cmpSparseIndices,
              oriBlockTable, cmpBlockTable, cuSeqlensQ, cuSeqlensOriKv, cuSeqlensCmpKv, sequsedQ, sequsedOriKv,
              sequsedCmpKv, cmpResidualKv, oriTopkLength, cmpTopkLength, sinks, metadata, quantMode, ropeHeadDim,
              softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQPtr, layoutKvPtr,
              topkValueMode, returnSoftmaxLse, attenOut, softmaxLse);
    return std::tuple<at::Tensor, at::Tensor>(attenOut, softmaxLse);
}

} // namespace op_api

namespace cann_ops_nn {
namespace quant {

constexpr int64_t N_CENT = 16;
constexpr int64_t SUPPORTED_HEAD_DIM = 512;

std::tuple<at::Tensor, at::Tensor> turbo_quant(const at::Tensor& latent, const at::Tensor& centroids)
{
    TORCH_CHECK(latent.is_meta() || latent.device().type() == at::kPrivateUse1, "latent must be on NPU device");
    TORCH_CHECK(centroids.is_meta() || centroids.device().type() == at::kPrivateUse1, "centroids must be on NPU device");

    TORCH_CHECK(latent.dim() == 2, "latent must be 2-dimensional [numTokens, headDim], but got ", latent.dim());
    TORCH_CHECK(latent.scalar_type() == at::kFloat,
                "latent dtype must be float32; float16 and bfloat16 are unsupported, but got ", latent.scalar_type());
    TORCH_CHECK(latent.is_contiguous(), "latent must be contiguous");
    TORCH_CHECK(centroids.scalar_type() == at::kFloat, "centroids dtype must be float32, but got ",
                centroids.scalar_type());
    TORCH_CHECK(centroids.is_contiguous(), "centroids must be contiguous");
    TORCH_CHECK(centroids.dim() == 1 && centroids.size(0) == N_CENT, "centroids must have shape [", N_CENT,
                "], but got ", centroids.sizes());

    const int64_t num_tokens = latent.size(0);
    const int64_t head_dim = latent.size(1);
    TORCH_CHECK(head_dim == SUPPORTED_HEAD_DIM, "headDim only supports ", SUPPORTED_HEAD_DIM, " for now, but got ",
                head_dim);

    at::Tensor y = at::empty({num_tokens, head_dim / 2}, at::TensorOptions().dtype(at::kByte).device(latent.device()));
    at::Tensor scale = at::empty({num_tokens}, at::TensorOptions().dtype(at::kHalf).device(latent.device()));

    if (latent.is_meta()) return {y, scale};
    EXEC_NPU_CMD(aclnnTurboQuant, latent, centroids, y, scale);
    return std::make_tuple(y, scale);
}

} // namespace quant
} // namespace cann_ops_nn


TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def(R"schema(mixed_quant_sparse_flash_mla(Tensor q,
        *,Tensor? ori_kv=None,
        Tensor? cmp_kv=None,
        Tensor? ori_sparse_indices=None,
        Tensor? cmp_sparse_indices=None,
        Tensor? ori_block_table=None,
        Tensor? cmp_block_table=None,
        Tensor? cu_seqlens_q=None,
        Tensor? cu_seqlens_ori_kv=None,
        Tensor? cu_seqlens_cmp_kv=None,
        Tensor? seqused_q=None,
        Tensor? seqused_ori_kv=None,
        Tensor? seqused_cmp_kv=None,
        Tensor? cmp_residual_kv=None,
        Tensor? ori_topk_length=None,
        Tensor? cmp_topk_length=None,
        Tensor? sinks=None,
        Tensor? metadata=None,
        int quant_mode=3,
        int rope_head_dim=64,
        float softmax_scale=1.0,
        int cmp_ratio=1,
        int ori_mask_mode=0,
        int cmp_mask_mode=0,
        int ori_win_left=-1,
        int ori_win_right=-1,
        str layout_q="BSND",
        str layout_kv="BSND",
        int topk_value_mode=1,
        bool return_softmax_lse=False) -> (Tensor,
        Tensor))schema");
    m.impl("mixed_quant_sparse_flash_mla", torch::kPrivateUse1, &op_api::MixedQuantSparseFlashMla);
    m.impl("mixed_quant_sparse_flash_mla", torch::kMeta, &op_api::MixedQuantSparseFlashMla);
    m.def(R"schema(turbo_quant(Tensor latent, Tensor centroids) -> (Tensor, Tensor))schema");
    m.impl("turbo_quant", torch::kPrivateUse1, &cann_ops_nn::quant::turbo_quant);
    m.impl("turbo_quant", torch::kMeta, &cann_ops_nn::quant::turbo_quant);
}
