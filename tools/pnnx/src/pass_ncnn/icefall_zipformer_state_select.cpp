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

class IcefallZipformerStateSelect : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
3 2
pnnx.Input                       x                0 1 x
zipformer.ZipformerStateSelect   op_0             1 1 x out i=%i
pnnx.Output                      output           1 0 out
)PNNXIR";
    }

    const char* type_str() const
    {
        return "Crop";
    }

    const char* name_str() const
    {
        return "crop";
    }

    void write(Operator* op, const std::map<std::string, Parameter>& captured_params) const
    {
        int i = captured_params.at("i").i;
        op->params["9"] = {i};      //start
        op->params["10"] = {i + 1}; // end
        op->params["11"] = {0};     // axis
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(IcefallZipformerStateSelect, 20)

} // namespace ncnn

} // namespace pnnx
