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

void insert_reshape_after_zipformer_state_select(Graph& graph)
{
    while (1)
    {
        bool matched = false;

        for (size_t i = 0; i < graph.ops.size(); i++)
        {
            Operator* op = graph.ops[i];

            if (op->type != "zipformer.ZipformerStateSelect")
                continue;

            Operand* op_out = op->outputs[0];
            if (op_out->consumers.size() == 1 && op_out->consumers[0]->type == "Reshape")
            {
                // We have already processed this op before, so skip it
                continue;
            }

            matched = true;

            Operator* reshape = graph.new_operator_after("Reshape", op->name + "_ncnnreshape0", op);
            // Change a 3-d tensor to a 2-d tensor by removing the channel axis
            // e.g., change (1, 384, 30) -> (384, 30)
            reshape->params["0"] = 0;
            reshape->params["1"] = 0;

            Operand* reshape_out = graph.new_operand(op->name + "_ncnnreshape0_out");
            reshape_out->producer = reshape;
            reshape->outputs.push_back(reshape_out);

            for (size_t j = 0; j < op_out->consumers.size(); ++j)
            {
                Operator* op2 = op_out->consumers[j];
                reshape_out->consumers.push_back(op2);

                for (size_t k = 0; k < op2->inputs.size(); ++k)
                {
                    if (op2->inputs[k] == op_out)
                    {
                        op2->inputs[k] = reshape_out;
                    }
                }
            }
            op_out->consumers.clear();
            op_out->consumers.push_back(reshape);
            reshape->inputs.push_back(op_out);

            break;
        }

        if (!matched)
            break;
    }
}

} // namespace ncnn

} // namespace pnnx
