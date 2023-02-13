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

class IcefallSimpleUpsample : public FuseModulePass
{
public:
    const char* match_type_str() const
    {
        return "__torch__.zipformer.SimpleUpsample";
    }

    const char* type_str() const
    {
        return "zipformer.SimpleUpsample";
    }

    virtual void write(Operator* op, const std::shared_ptr<torch::jit::Graph>& graph, const torch::jit::Module& mod) const
    {
        int upsample = mod.attr("upsample").toInt();
        int num_channels = mod.attr("num_channels").toInt();

        const auto& bias = mod.attr("bias").toTensor();
        if (bias.size(0) != upsample)
        {
            fprintf(stderr, "bias.size(0) %d vs %d\n", (int)bias.size(0), upsample);
            exit(-1);
        }

        if (bias.size(1) != num_channels)
        {
            fprintf(stderr, "bias.size(1) %d vs %d\n", (int)bias.size(1), num_channels);
            exit(-1);
        }

        op->params["upsample"] = upsample;
        op->params["num_channels"] = num_channels;

        op->attrs["bias"] = bias;
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(IcefallSimpleUpsample)

class IcefallSimpleUpsample_2 : public IcefallSimpleUpsample
{
public:
    const char* match_type_str() const
    {
        return "__torch__.zipformer2.SimpleUpsample";
    }
};

REGISTER_GLOBAL_PNNX_FUSE_MODULE_PASS(IcefallSimpleUpsample_2)

} // namespace pnnx
