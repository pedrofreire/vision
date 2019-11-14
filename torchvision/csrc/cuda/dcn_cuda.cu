#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include "cuda_helpers.h"

#include <iostream>


using namespace at;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N)
{
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(const scalar_t *in, const int height, const int width, scalar_t h, scalar_t w)
{
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = in[h_low * width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = in[h_low * width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = in[h_high * width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = in[h_high * width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__global__ void deformable_im2col_gpu_kernel(const int n, const scalar_t* input_ptr, const scalar_t* offset_ptr,
                                             const int height, const int width, const int weight_h, const int weight_w,
                                             const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                             const int dil_h, const int dil_w,
                                             const int batch_sz, const int n_in_channels, const int deformable_group,
                                             const int out_h, const int out_w,
                                             scalar_t* columns_ptr)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int out_b = (index / (out_w * out_h)) % batch_sz;
    const int in_c = index / (out_w * out_h * batch_sz);
    const int out_c = in_c * weight_h * weight_w;

    int channel_per_deformable_group = n_in_channels / deformable_group;
    const int grp_idx = in_c / channel_per_deformable_group;

    columns_ptr += (out_c * (batch_sz * out_h * out_w)
                  + out_b * (out_h * out_w)
                  + out_y * out_w
                  + out_x);

    input_ptr += (out_b * (n_in_channels * height * width)
                 + in_c * (height * width));

    offset_ptr += (out_b * deformable_group + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const int offset_idx = 2 * (i * weight_w + j);
        const scalar_t offset_h = offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const scalar_t offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
        const scalar_t y = (out_y * stride_h - pad_h) + i * dil_h + offset_h;
        const scalar_t x = (out_x * stride_w - pad_w) + j * dil_w + offset_w;
        *columns_ptr = bilinear_interpolate(input_ptr, height, width, y, x);
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}

void deformable_im2col(
    const at::Tensor input, const at::Tensor data_offset, int n_in_channels,
    int height, int width,
    int weight_h, int weight_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dil_h, int dil_w,
    int out_h, int out_w,
    int parallel_imgs, int deformable_group, at::Tensor data_col) {
  int num_kernels = n_in_channels * out_h * out_w * parallel_imgs;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "deformable_im2col_gpu", ([&] {
        deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            input.data_ptr<scalar_t>(),
            data_offset.data_ptr<scalar_t>(),
            height, width, weight_h, weight_w, pad_h, pad_w, stride_h, stride_w, dil_h, dil_w,
            parallel_imgs, n_in_channels, deformable_group,
            out_h, out_w,
            data_col.data_ptr<scalar_t>());
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
  }
}

void shape_check(at::Tensor input, at::Tensor offset, at::Tensor *gradOutput,
                 at::Tensor weight, std::pair<int, int> stride, std::pair<int, int> pad,
                 std::pair<int, int> dilation, int n_weight_grps, int n_offset_grps) {
  TORCH_CHECK(input.ndimension() == 4);
  TORCH_CHECK(offset.ndimension() == 4);
  TORCH_CHECK(weight.ndimension() == 4);
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(offset.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());

  int in_h = input.size(2);
  int in_w = input.size(3);

  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  int stride_h = stride.first;
  int stride_w = stride.second;

  int pad_h = pad.first;
  int pad_w = pad.second;

  int dil_h = dilation.first;
  int dil_w = dilation.second;

  int ker_h = dil_h * (weight_h - 1) + 1;
  int ker_w = dil_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2*pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2*pad_w - ker_w) / stride_w) + 1;

  TORCH_CHECK(weight_h > 0 && weight_w > 0);
  TORCH_CHECK(stride_h > 0 && stride_w > 0);
  TORCH_CHECK(dil_h > 0 && dil_w > 0, "dil_h: ", dil_w, " dil_w: ", dil_h);
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "pad_h: ", pad_w, " pad_w: ", pad_h);

  TORCH_CHECK(weight.size(1) * n_weight_grps == input.size(1));
  TORCH_CHECK(weight.size(0) % n_weight_grps == 0);
  TORCH_CHECK(input.size(1) % n_offset_grps == 0);

  TORCH_CHECK((offset.size(0) == input.size(0)), "invalid batch size of offset");
  TORCH_CHECK((offset.size(1) == n_offset_grps * 2 * weight_h * weight_w),
           "invalid number of channels of offset");
  TORCH_CHECK((offset.size(2) == out_h && offset.size(3) == out_w),
           "offset output dims: (", offset.size(2), ", ", offset.size(3),
           ") - output dims: (", out_h, ", ", out_w, ")");

  TORCH_CHECK(out_h > 0 && out_w > 0,
      "Calculated output size too small - out_h: ", out_h, " out_w: ", out_w);

  if (gradOutput != NULL) {
  }
}


at::Tensor DCN_forward_cuda(
    at::Tensor input,
    at::Tensor offset,
    at::Tensor weight,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int n_weight_grps, int n_offset_grps, int im2col_block) {

  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");
  int batch_size = input.size(0);
  im2col_block = std::min(batch_size, im2col_block);
  TORCH_CHECK(batch_size % im2col_block == 0);
  shape_check(input, offset, NULL, weight, stride, pad, dilation, n_weight_grps, n_offset_grps);

  at::DeviceGuard guard(input.device());
  
  // make args contiguous
  input = input.contiguous();
  offset = offset.contiguous();
  weight = weight.contiguous();

  // Unpack shapes and args
  int out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  int stride_h = stride.first;
  int stride_w = stride.second;

  int pad_h = pad.first;
  int pad_w = pad.second;

  int dil_h = dilation.first;
  int dil_w = dilation.second;

  int batch_sz = input.size(0);
  int in_channels = input.size(1);
  int in_h = input.size(2);
  int in_w = input.size(3);

  // Initialize output tensor
  int ker_h = dil_h * (weight_h - 1) + 1;
  int ker_w = dil_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2*pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2*pad_w - ker_w) / stride_w) + 1;

  auto out = at::zeros({batch_sz, out_channels, out_h, out_w}, input.options());

  // Separate batches into blocks
  out = out.view({batch_sz / im2col_block, im2col_block, out_channels, out_h, out_w});
  input = input.view({batch_sz / im2col_block, im2col_block, in_channels, in_h, in_w});
  offset = offset.view({batch_sz / im2col_block, im2col_block, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});
  at::Tensor out_buf = at::zeros({batch_sz / im2col_block, out_channels, im2col_block * out_h, out_w}, out.options());

  // Separate channels into convolution groups
  out_buf = out_buf.view({out_buf.size(0), n_weight_grps, out_buf.size(1) / n_weight_grps, out_buf.size(2), out_buf.size(3)}); 
  weight = weight.view({n_weight_grps, weight.size(0) / n_weight_grps, weight.size(1), weight.size(2), weight.size(3)});

  // Sample points and perform convolution
  auto columns = at::zeros({in_channels * weight_h * weight_w, im2col_block * out_h * out_w}, input.options());
  for (int b = 0; b < batch_sz / im2col_block; b++) {
    deformable_im2col(input[b], offset[b], in_channels, in_h,
                      in_w, weight_w, weight_h, pad_h, pad_w, stride_h, stride_w, dil_h,
                      dil_w, out_h, out_w, im2col_block, n_offset_grps, columns);

    columns = columns.view({n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    for (int g = 0; g < n_weight_grps; g++) {
      out_buf[b][g] = out_buf[b][g].flatten(1)
                                   .addmm_(weight[g].flatten(1), columns[g])
                                   .view_as(out_buf[b][g]);
    }
  }

  out_buf = out_buf.view({batch_sz / im2col_block, out_channels, im2col_block, out_h, out_w});
  out_buf.transpose_(1, 2);
  out.copy_(out_buf);
  out = out.view({batch_sz, out_channels, out_h, out_w});

  return out;
}


template <typename scalar_t>
__device__ scalar_t get_gradient_weight(scalar_t y, scalar_t x,
                                        const int yp, const int xp, const int height, const int width) {
  if (y <= -1 || y >= height || x <= -1 || x >= width) {
    return 0;
  }

  return (1 - abs(y - yp)) * (1 - abs(x - xp));

  int y_low = floor(y);
  int x_low = floor(x);
  int y_high = y_low + 1;
  int x_high = x_low + 1;

  scalar_t weight = 0;
  if (yp == y_low && xp == x_low)
    weight = (yp + 1 - y) * (xp + 1 - x);
  if (yp == y_low && xp == x_high)
    weight = (yp + 1 - y) * (x + 1 - xp);
  if (yp == y_high && xp == x_low)
    weight = (y + 1 - yp) * (xp + 1 - x);
  if (yp == y_high && xp == x_high)
    weight = (y + 1 - yp) * (x + 1 - xp);
  return weight;
}

template <typename scalar_t>
__device__ scalar_t get_coordinate_weight(scalar_t argmax_h, scalar_t argmax_w,
                                          const int height, const int width, const scalar_t *im_data,
                                          const int data_width, const int bp_dir)
{

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  scalar_t weight = 0;

  if (bp_dir == 0)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }
  else if (bp_dir == 1)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}


template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(
    const int n, const scalar_t *col, const scalar_t *offset_ptr,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int n_offset_grps,
    const int out_h, const int out_w,
    scalar_t *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int j = (index / (out_w * out_h * batch_size)) % kernel_w;
    const int i = (index / (out_w * out_h * batch_size * kernel_w)) % kernel_h;
    const int c = index / (out_w * out_h * batch_size * kernel_w * kernel_h);
    // compute the start and end of the output

    const int offset_grp = c / channel_per_deformable_group;

    int out_x = index % out_w;
    int out_y = (index / out_w) % out_h;
    int b = (index / (out_w * out_h)) % batch_size;

    offset_ptr += (b * n_offset_grps + offset_grp) * 2 * kernel_h * kernel_w * out_h * out_w;
    const int offset_h_ptr = ((2 * (i * kernel_w + j)) * out_h + out_y) * out_w + out_x;
    const int offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * out_h + out_y) * out_w + out_x;
    const scalar_t offset_h = offset_ptr[offset_h_ptr];
    const scalar_t offset_w = offset_ptr[offset_w_ptr];
    const scalar_t y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
    const scalar_t x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

    const int cur_h = (int)y;
    const int cur_w = (int)x;
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(y - (cur_h + dy)) < 1 &&
            abs(x - (cur_w + dx)) < 1) {
          int grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          scalar_t weight = get_gradient_weight(y, x, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + grad_pos, weight * col[index]);
        }
      }
    }
  }
}

void deformable_col2im(
    const at::Tensor columns, const at::Tensor offset, const int channels,
    const int height, const int width, const int weight_h,
    const int weight_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int n_offset_grps,
    at::Tensor grad_im) {
  int out_h = (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
  int out_w = (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * weight_h * weight_w * out_h * out_w * parallel_imgs;
  int channel_per_deformable_group = channels / n_offset_grps;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      columns.scalar_type(), "deformable_col2im_gpu", ([&] {
        deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            columns.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(),
            channels, height, width, weight_h,
            weight_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group,
            parallel_imgs, n_offset_grps, out_h, out_w,
            grad_im.data_ptr<scalar_t>());
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(const int n, const scalar_t *data_col_ptr,
                                                   const scalar_t *data_im_ptr, const scalar_t *data_offset_ptr,
                                                   const int channels, const int height, const int width,
                                                   const int weight_h, const int weight_w,
                                                   const int pad_h, const int pad_w,
                                                   const int stride_h, const int stride_w,
                                                   const int dilation_h, const int dilation_w,
                                                   const int channel_per_deformable_group,
                                                   const int batch_size, const int offset_channels, const int n_offset_grps,
                                                   const int out_h, const int out_w, scalar_t *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    scalar_t val = 0;
    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int c = (index / (out_w * out_h)) % offset_channels;
    int b = index / (out_w * out_h * offset_channels);
    // compute the start and end of the output

    const int offset_grp = c / (2 * weight_h * weight_w);
    const int col_step = weight_h * weight_w;
    int cnt = 0;

    data_col_ptr += offset_grp * channel_per_deformable_group * batch_size * out_w * out_h;
    data_im_ptr += (b * n_offset_grps + offset_grp) * channel_per_deformable_group / weight_h / weight_w * height * width;
    data_offset_ptr += (b * n_offset_grps + offset_grp) * 2 * weight_h * weight_w * out_h * out_w;

    const int offset_c = c - offset_grp * 2 * weight_h * weight_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) * out_h) + h) * out_w + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / out_w / out_h / batch_size) % weight_w;
      int i = (col_pos / out_w / out_h / batch_size / weight_w) % weight_h;
      int out_x = col_pos % out_w;
      int out_y = (col_pos / out_w) % out_h;
      int in_x = out_x * stride_w - pad_w;
      int in_y = out_y * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * weight_w + j)) * out_h + out_y) * out_w + out_x);
      const int data_offset_w_ptr = (((2 * (i * weight_w + j) + 1) * out_h + out_y) * out_w + out_x);
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_h = in_y + i * dilation_h + offset_h;
      scalar_t inv_w = in_x + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
      {
        inv_h = inv_w = -2;
      }
      const scalar_t weight = get_coordinate_weight(
          inv_h, inv_w,
          height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
  }
}

void deformable_col2im_coord(
    const at::Tensor columns, const at::Tensor input, const at::Tensor offset,
    const int channels, const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int n_offset_grps, at::Tensor grad_offset) {
  int out_h = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int out_w = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = out_h * out_w * 2 * ksize_h * ksize_w * n_offset_grps * parallel_imgs;
  int channel_per_deformable_group = channels * ksize_h * ksize_w / n_offset_grps;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      columns.scalar_type(), "deformable_col2im_coord_gpu", ([&] {
        deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels,
            columns.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(),
            channels, height, width, ksize_h,
            ksize_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group,
            parallel_imgs, 2 * ksize_h * ksize_w * n_offset_grps, n_offset_grps,
            out_h, out_w,
            grad_offset.data_ptr<scalar_t>());
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_col2im_coord: %s\n", cudaGetErrorString(err));
  }
}


std::tuple<at::Tensor, at::Tensor> deform_conv_backward_input_cuda(
    at::Tensor input, at::Tensor offset, at::Tensor weight,
    at::Tensor grad_out,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int n_weight_grps, int n_offset_grps, int im2col_block) {

  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  int stride_h = stride.first;
  int stride_w = stride.second;

  int pad_h = pad.first;
  int pad_w = pad.second;

  int dil_h = dilation.first;
  int dil_w = dilation.second;

  long batch_sz = input.size(0);
  long n_in_channels = input.size(1);
  long in_h = input.size(2);
  long in_w = input.size(3);

  long n_out_channels = weight.size(0);

  long out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) / stride_w + 1;
  long out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) / stride_h + 1;

  at::DeviceGuard guard(input.device());

  auto grad_input = at::zeros_like(input);
  auto grad_offset = at::zeros_like(offset);
  auto columns = at::zeros({n_in_channels * weight_w * weight_h, im2col_block * out_h * out_w}, input.options());

  // Separate into blocks

  std::cout << "backward_input: \n";
  std::cout << "input.sizes(): " << input.sizes() << "\n";
  std::cout << "offset.sizes(): " << offset.sizes() << "\n";
  std::cout << "weight.sizes(): " << weight.sizes() << "\n";
  std::cout << "grad_out.sizes(): " << grad_out.sizes() << "\n";
  std::cout << "columns.sizes(): " << columns.sizes() << "\n";

  std::cout << "A\n";
  grad_input = grad_input.view({batch_sz / im2col_block, im2col_block, n_in_channels, in_h, in_w});
  input = input.view({batch_sz / im2col_block, im2col_block, n_in_channels, in_h, in_w});
  grad_offset = grad_offset.view({batch_sz / im2col_block, im2col_block, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});
  offset = offset.view({batch_sz / im2col_block, im2col_block, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  std::cout << "B\n";
  grad_out = grad_out.view({batch_sz / im2col_block, im2col_block, n_out_channels, out_h, out_w});
  grad_out.transpose_(1, 2);
  std::cout << "C\n";
  grad_out = grad_out.view(
      {grad_out.size(0), n_weight_grps, grad_out.size(1) / n_weight_grps,
       grad_out.size(2), grad_out.size(3), grad_out.size(4)});
  std::cout << "D\n";

  for (int elt = 0; elt < batch_sz / im2col_block; elt++) {
    // Separate into weight groups
    columns = columns.view({n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    weight = weight.view({n_weight_grps, weight.size(0) / n_weight_grps, weight.size(1), weight.size(2), weight.size(3)});
    for (int g = 0; g < n_weight_grps; g++) {
      columns[g] = columns[g].addmm_(weight[g].flatten(1).transpose(0, 1), grad_out[elt][g].flatten(1));
    }
    columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});

    deformable_col2im_coord(columns, input[elt], offset[elt], n_in_channels,
                            in_h, in_w, weight_h, weight_w, pad_h, pad_w, stride_h, stride_w,
                            dil_h, dil_w, im2col_block, n_offset_grps,
                            grad_offset[elt]);

    deformable_col2im(columns, offset[elt], n_in_channels, in_h,
                      in_w, weight_h, weight_w, pad_h, pad_w, stride_h, stride_w, dil_h,
                      dil_w, im2col_block, n_offset_grps, grad_input[elt]);
  }

  std::cout << "E\n";
  grad_out = grad_out.view(
      {grad_out.size(0), grad_out.size(1) * grad_out.size(2),
       grad_out.size(3), grad_out.size(4), grad_out.size(5)});
  std::cout << "F\n";
  grad_out.transpose_(1, 2);
  grad_out = grad_out.view({batch_sz, n_out_channels, out_h, out_w});
  std::cout << "G\n";

  grad_input = grad_input.view({batch_sz, n_in_channels, in_h, in_w});
  input = input.view({batch_sz, n_in_channels, in_h, in_w});
  grad_offset = grad_offset.view({batch_sz, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});
  offset = offset.view({batch_sz, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  return {grad_input, grad_offset};
}



at::Tensor deform_conv_backward_parameters_cuda(
    at::Tensor input, at::Tensor offset, at::Tensor weight,
    at::Tensor grad_out,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int n_weight_grps, int n_offset_grps, int im2col_block) {
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  int stride_h = stride.first;
  int stride_w = stride.second;

  int pad_h = pad.first;
  int pad_w = pad.second;

  int dil_h = dilation.first;
  int dil_w = dilation.second;

  at::DeviceGuard guard(input.device());

  long batch_sz = input.size(0);
  long n_in_channels = input.size(1);
  long in_h = input.size(2);
  long in_w = input.size(3);

  long n_out_channels = weight.size(0);

  long out_w = grad_out.size(2);
  long out_h = grad_out.size(3);

  auto grad_weight = at::zeros_like(weight);;
  auto columns = at::zeros({n_in_channels * weight_w * weight_h, im2col_block * out_h * out_w}, input.options());

  std::cout << "backward_parameters: \n";
  std::cout << "input.sizes(): " << input.sizes() << "\n";
  std::cout << "offset.sizes(): " << offset.sizes() << "\n";
  std::cout << "weight.sizes(): " << weight.sizes() << "\n";
  std::cout << "grad_out.sizes(): " << grad_out.sizes() << "\n";
  std::cout << "columns.sizes(): " << columns.sizes() << "\n";

  std::cout << "J\n";
  grad_out = grad_out.view({batch_sz / im2col_block, im2col_block,
                                n_out_channels, out_h, out_w});
  grad_out.transpose_(1, 2);

  at::Tensor grad_out_buf = at::zeros_like(grad_out);
  grad_out_buf.copy_(grad_out);
  std::cout << "K\n";
  grad_out_buf = grad_out_buf.view({grad_out_buf.size(0), n_weight_grps, grad_out_buf.size(1) / n_weight_grps, grad_out_buf.size(2), grad_out_buf.size(3)});

  std::cout << "L\n";
  grad_out.transpose_(1, 2);
  grad_out = grad_out.view({batch_sz, n_out_channels, out_h, out_w});
  std::cout << "M\n";

  input = input.view({batch_sz / im2col_block, im2col_block, n_in_channels, in_h, in_w});
  offset = offset.view({batch_sz / im2col_block, im2col_block, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  grad_weight = grad_weight.view({n_weight_grps, grad_weight.size(0) / n_weight_grps, grad_weight.size(1), grad_weight.size(2), grad_weight.size(3)});
  for (int elt = 0; elt < batch_sz / im2col_block; elt++) {
    deformable_im2col(input[elt], offset[elt], n_in_channels, in_h,
                      in_w, weight_h, weight_w, pad_h, pad_w, stride_h, stride_w, dil_h,
                      dil_w, im2col_block, out_h, out_w, n_offset_grps, columns);

    columns = columns.view({n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    for (int g = 0; g < n_weight_grps; g++) {
      grad_weight[g] = grad_weight[g]
                          .flatten(1)
                          .addmm_(grad_out_buf[elt][g].flatten(1), columns[g].transpose(1, 0))
                          .view_as(grad_weight[g]);
    }
    columns = columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  std::cout << "N\n";
  input = input.view({batch_sz, n_in_channels, in_h, in_w});
  offset = offset.view({batch_sz, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  grad_weight = grad_weight.view({grad_weight.size(0) * grad_weight.size(1),
                                grad_weight.size(2), grad_weight.size(3), grad_weight.size(4)});
  return grad_weight;
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> DCN_backward_cuda(
    at::Tensor grad_out,
    at::Tensor input,
    at::Tensor offset,
    at::Tensor weight,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int groups,
    int deformable_groups,
    int im2col_step) {

  auto grad_input_and_offset = deform_conv_backward_input_cuda(
      input, offset, weight, grad_out,
      stride, pad, dilation,
      groups, deformable_groups, im2col_step);

  auto grad_input = std::get<0>(grad_input_and_offset);
  auto grad_offset = std::get<1>(grad_input_and_offset);

  auto grad_weight = deform_conv_backward_parameters_cuda(
      input, offset, weight, grad_out,
      stride, pad, dilation,
      groups, deformable_groups, im2col_step);

  return {grad_input, grad_offset, grad_weight};
}


