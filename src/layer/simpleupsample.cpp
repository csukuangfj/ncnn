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

#include "simpleupsample.h"

namespace ncnn {

SimpleUpsample::SimpleUpsample()
{
    one_blob_only = true;
    support_inplace = false;
}

int SimpleUpsample::load_param(const ParamDict& pd)
{
    upsample = pd.get(0, 0);
    num_channels = pd.get(1, 0);
    int bias_data_size = pd.get(2, 0);
    if (bias_data_size != upsample * num_channels)
    {
        NCNN_LOGE("upsample: %d, num_channels: %d, bias_data_size: %d. %dx%d!=%d", upsample, num_channels, bias_data_size,
                  upsample, num_channels, bias_data_size);
        return -100;
    }

    return 0;
}

int SimpleUpsample::load_model(const ModelBin& mb)
{
    bias = mb.load(num_channels, upsample, 0);
    if (bias.empty())
        return -100;

    return 0;
}

int SimpleUpsample::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // bottom_blob.dims == 2
    // bottom_blob.w == seq_len
    // bottom_blob.h == num_channels

    int outw = bottom_blob.w;
    int outh = upsample;
    int outc = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < outc; ++q)
    {
        Mat out_m = top_blob.channel(q);
        const float* a_ptr = bottom_blob.row(q);

        for (int y = 0; y < outh; ++y)
        {
            float* out_ptr = out_m.row(y);
            const float* b_ptr = bias.row(y);
            for (int x = 0; x < outw; ++x)
            {
                out_ptr[x] = a_ptr[x] + b_ptr[x];
            }
        }
    }

    top_blob = top_blob.reshape(outw, outh * outc);

    return 0;
}

} // namespace ncnn
