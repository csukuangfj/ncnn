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

#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

class IcefallApplyPaddingMask: public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        printf("Hello from ncnn apply_padding_mask match\n");
        return R"PNNXIR(7767517
4 3
pnnx.Input                                       x                0 1 x
pnnx.Input                                       a                0 1 a
emformer2.ApplyPaddingMask                       op_0             2 1 x a out nhead=%nhead
pnnx.Output                                      output           1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        printf("Hello from ncnn apply_padding_mask type_str\n");
        return "IcefallApplyPaddingMask";
    }

    const char* name_str() const
    {
        printf("Hello from ncnn apply_padding_mask name_str\n");
        return "icefall_apply_paddingmask";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int nhead = captured_params.at("nhead").i;
        op->params["0"] = nhead;
        printf("Hello from ncnn apply_padding_mask write\n");
    }
};


REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(IcefallApplyPaddingMask, 20)

} // namespace ncnn

} // namespace pnnx
