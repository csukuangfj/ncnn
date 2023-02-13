// Copyright (c) 2023 Xiaomi Corp.        (author: Fangjun Kuang)
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "tensorasstrided.h"

namespace ncnn {

TensorAsStrided::TensorAsStrided()
{
    one_blob_only = true;
    support_inplace = false;
}

int TensorAsStrided::load_param(const ParamDict& pd)
{
    sizes = pd.get(0, Mat());
    strides = pd.get(1, Mat());
    storage_offset = pd.get(2, 0);

    if (sizes.dims != 1 && strides.dims != 1)
    {
        NCNN_LOGE("sizes.dims: %d, strides.dims: %d. They are not 1!\n", sizes.dims, strides.dims);
        return -100;
    }

    if (sizes.w != strides.w)
    {
        NCNN_LOGE("sizes.w: %d, strides.w: %d. They are not equal!\n", sizes.w, strides.w);
        return -100;
    }

    return 0;
}

int TensorAsStrided::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int* p_sizes = sizes;
    const int* p_strides = strides;

    if (sizes.w == 3)
    {
        if (bottom_blob.dims != 3)
        {
            NCNN_LOGE("Only 3-D tensors are supported right now");
            return -100;
        }

        int inc = bottom_blob.c;
        int inh = bottom_blob.h;
        int inw = bottom_blob.w;

        int outc = p_sizes[0];
        int outh = p_sizes[1];
        int outw = p_sizes[2];

        if (bottom_blob.c != outc)
        {
            NCNN_LOGE("We only implement in_c == out_c right now");
            return -100;
        }

        if (p_strides[0] != inh * inw)
        {
            NCNN_LOGE("Stride that crosses channels is not supported");
            return -100;
        }

        size_t elemsize = bottom_blob.elemsize;
        top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);

        int stride1 = p_strides[1];
        int stride2 = p_strides[2];

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < outc; q++)
        {
            Mat out_m = top_blob.channel(q);

            const float* in_m = bottom_blob.channel(q);
            in_m += storage_offset;

            for (int y = 0; y < outh; ++y)
            {
                float* out_ptr = out_m.row(y);
                const float* in_ptr = in_m + y * stride1;
                for (int x = 0; x < outw; ++x)
                {
                    out_ptr[x] = in_ptr[x * stride2];
                }
            }
        }

        return 0;
    }

    NCNN_LOGE("TensorAsStrided: Only 3-D tensors are supported right now");

    return -100;
}

} // namespace ncnn
