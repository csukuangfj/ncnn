// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//               2023 Xiaomi Corp.        (author: Fangjun Kuang)
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pass_level1.h"

#include "../utils.h"

namespace pnnx {

class IcefallRelPositionMultiheadAttentionPermute : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.zipformer.RelPositionMultiheadAttentionPermute";
    }

    const char* type_str() const
    {
        return "zipformer.RelPositionMultiheadAttentionPermute";
    }

    virtual void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        op->params["kind"] = mod.attr("kind").toInt();
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(IcefallRelPositionMultiheadAttentionPermute)

class IcefallRelPositionMultiheadAttentionPermute_2 : public IcefallRelPositionMultiheadAttentionPermute
{
public:
    const char* match_type_str() const
    {
        return "__torch__.zipformer2.RelPositionMultiheadAttentionPermute";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(IcefallRelPositionMultiheadAttentionPermute_2)

} // namespace pnnx
