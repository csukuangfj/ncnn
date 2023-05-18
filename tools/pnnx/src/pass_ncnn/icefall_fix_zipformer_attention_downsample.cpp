// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "insert_reshape_linear.h"
#include "pass_ncnn.h"

namespace pnnx {

namespace ncnn {

static bool ends_with(std::string const& str, std::string const& suffix)
{
    if (suffix.size() > str.size())
        return false;

    return 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

void icefall_fix_zipformer_attention_downsample(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "MemoryData")
                continue;

            if (!ends_with(op->name, "downsample"))
                continue;

            if (op->params.find("2") != op->params.end())
                continue;

            // ncnn ignores the batch index for the query version
            // in AttenionDownsample; we give it an extra dim here.
            op->params["1"] = 1;
            op->params["2"] = 1;

            // find the consumer of MemoryData, which is a BinaryOp
            Operator* binary_op = op->outputs[0]->consumers[0];

            if (binary_op->type != "BinaryOp")
                continue;

            Operator* reduction_op = binary_op->outputs[0]->consumers[0];
            if (reduction_op->type != "Reduction")
                continue;

            Operator* softmax_op = reduction_op->outputs[0]->consumers[0];

            if (softmax_op->type != "Softmax")
                continue;

            softmax_op->params["0"] = 1;
            softmax_op->params["1"] = 1;

            binary_op = softmax_op->outputs[0]->consumers[0];
            if (binary_op->type != "BinaryOp")
                continue;

            reduction_op = binary_op->outputs[0]->consumers[0];
            if (reduction_op->type != "Reduction")
                continue;

            reduction_op->params["3"] = {1};

            matched = true;

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
