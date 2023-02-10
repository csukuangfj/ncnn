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

class IcefallComputePaddingMask: public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        printf("Hello from ncnn match\n");
        return R"PNNXIR(7767517
5 4
pnnx.Input                                       x                0 1 x
pnnx.Input                                       a                0 1 a
pnnx.Input                                       b                0 1 b
emformer2.ComputePaddingMask                     op_0             3 1 x a b out left_context_length=%left_context_length right_context_length=%right_context_length memory_size=%memory_size shift=%shift
pnnx.Output                                      output           1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        printf("Hello from ncnn type_str\n");
        return "IcefallComputePaddingMask";
    }

    const char* name_str() const
    {
        printf("Hello from ncnn name_str\n");
        return "icefall_compute_paddingmask";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int right_context_length = captured_params.at("right_context_length").i;
        int left_context_length = captured_params.at("left_context_length").i;
        int shift = captured_params.at("shift").i;
        int memory_size = captured_params.at("memory_size").i;
        op->params["0"] = right_context_length;
        op->params["1"] = left_context_length;
        op->params["2"] = shift;
        op->params["3"] = memory_size;

        printf("Hello from ncnn\n");
    }
};


REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(IcefallComputePaddingMask, 20)

} // namespace ncnn

} // namespace pnnx
