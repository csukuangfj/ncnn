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

class Tensor_as_strided : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input     0 1 input
Tensor.as_strided       op        1 1 input out size=%size stride=%stride storage_offset=%storage_offset
pnnx.Output             output    1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "TensorAsStrided";
    }

    const char* name_str() const
    {
        return "tensor_as_strided";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        std::vector<int> size = captured_params.at("size").ai;
        std::vector<int> stride = captured_params.at("stride").ai;
        int storage_offset = captured_params.at("storage_offset").i;

        op->params["0"] = size;
        op->params["1"] = stride;
        op->params["2"] = storage_offset;
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(Tensor_as_strided, 20)

} // namespace ncnn

} // namespace pnnx
