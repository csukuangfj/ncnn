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

#include "poolingmodulenoproj.h"

namespace ncnn {

PoolingModuleNoProj::PoolingModuleNoProj()
{
    one_blob_only = false;
    support_inplace = false;
}

int PoolingModuleNoProj::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    Mat x = bottom_blobs[0];
    Mat cached_len = bottom_blobs[1];
    Mat cached_avg = bottom_blobs[2];

    // x.dims = 2, x.w = C, x.h = T
    // cached_len.dims = 1, cached_len.w = 1
    // cached_avg.dims = 2, cached_avg.w = C, cached_avg.h = 1

    Mat& out_x = top_blobs[0];
    out_x.create_like(x, opt.blob_allocator);

    Mat& out_cached_len = top_blobs[1];
    out_cached_len.create_like(cached_len, opt.blob_allocator);

    Mat& out_cached_avg = top_blobs[2];
    out_cached_avg.create_like(cached_avg, opt.blob_allocator);

    int w = x.w;
    int h = x.h;

    const float* x_ptr = x;
    const float* cached_avg_ptr = cached_avg;
    float* out_ptr = out_x;

    float n = cached_len[0];

    // process row 0
    for (int c = 0; c < w; ++c)
    {
        out_ptr[c] = x_ptr[c] + n * cached_avg_ptr[c];
    }

    for (int r = 1; r < h; ++r)
    {
        const float* x_cur = x.row(r);

        float* out_prev = out_x.row(r - 1);
        float* out_cur = out_x.row(r);

        float scale = 1. / (n + r); // scale for the previous row
        for (int c = 0; c < w; ++c)
        {
            out_cur[c] = out_prev[c] + x_cur[c];
            out_prev[c] *= scale;
        }
    }

    float* last_row = out_x.row(h - 1);
    float scale = 1. / (n + h);

    float* out_cached_avg_ptr = out_cached_avg;
    for (int c = 0; c < w; ++c)
    {
        last_row[c] *= scale;
        out_cached_avg_ptr[c] = last_row[c];
    }

    out_cached_len[0] = n + h;

    return 0;
}

} // namespace ncnn
