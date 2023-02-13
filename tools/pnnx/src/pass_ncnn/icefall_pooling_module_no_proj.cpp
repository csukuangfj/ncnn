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

class IcefallPoolingModuleNoProj : public GraphRewriterPass
{
public:
    const char* match_pattern_graph() const
    {
        return R"PNNXIR(7767517
5 6
pnnx.Input                              x                0 1 x
pnnx.Input                              cached_len       0 1 cached_len
pnnx.Input                              cached_avg       0 1 cached_avg
scaling_converter.PoolingModuleNoProj   op_0             3 3 x cached_len cached_avg out_x out_cached_len out_cached_avg
pnnx.Output                             output           3 0 out_x out_cached_len out_cached_avg
)PNNXIR";
    }

    const char* type_str() const
    {
        return "PoolingModuleNoProj";
    }

    const char* name_str() const
    {
        return "pooling_modul_no_proj";
    }
};

REGISTER_GLOBAL_PNNX_NCNN_GRAPH_REWRITER_PASS(IcefallPoolingModuleNoProj, 20)

} // namespace ncnn

} // namespace pnnx
