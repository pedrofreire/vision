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
__device__ scalar_t deformable_im2col_bilinear(const scalar_t *bottom_data, const int data_width,
                                               const int height, const int width, scalar_t h, scalar_t w)
{

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
__device__ scalar_t get_gradient_weight(scalar_t argmax_h, scalar_t argmax_w,
                                        const int h, const int w, const int height, const int width)
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
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
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
__global__ void deformable_im2col_gpu_kernel(const int n, const scalar_t *data_im, const scalar_t *data_offset,
                                             const int height, const int width, const int kernel_h, const int kernel_w,
                                             const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                                             const int dilation_h, const int dilation_w, const int channel_per_deformable_group,
                                             const int batch_size, const int num_channels, const int deformable_group,
                                             const int height_col, const int width_col,
                                             scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    scalar_t *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    //const scalar_t* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const scalar_t *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
        const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
        scalar_t val = static_cast<scalar_t>(0);
        const scalar_t h_im = h_in + i * dilation_h + offset_h;
        const scalar_t w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          //const scalar_t map_h = i * dilation_h + offset_h;
          //const scalar_t map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = deformable_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = deformable_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

void deformable_im2col(
    const at::Tensor data_im, const at::Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deformable_group, at::Tensor data_col)
{
  // num_axes should be smaller than block size
  // todo: check parallel_imgs is correctly passed in
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "deformable_im2col_gpu", ([&] {
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        scalar_t *data_col_ = data_col.data<scalar_t>();

        deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_im_, data_offset_, height, width, ksize_h, ksize_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            channel_per_deformable_group, parallel_imgs, channels, deformable_group,
            height_col, width_col, data_col_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_gpu_kernel(
    const int n, const scalar_t *data_col, const scalar_t *data_offset,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, const int deformable_group,
    const int height_col, const int width_col,
    scalar_t *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) *
                                                        2 * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
    const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
    const scalar_t cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const scalar_t cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const scalar_t cur_top_grad = data_col[index];
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          scalar_t weight = get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

void deformable_col2im(
    const at::Tensor data_col, const at::Tensor data_offset, const int channels,
    const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group,
    at::Tensor grad_im)
{

  // todo: make sure parallel_imgs is passed in correctly
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * ksize_h * ksize_w * height_col * width_col * parallel_imgs;
  int channel_per_deformable_group = channels / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        scalar_t *grad_im_ = grad_im.data<scalar_t>();

        deformable_col2im_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_col_, data_offset_, channels, height, width, ksize_h,
            ksize_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group,
            parallel_imgs, deformable_group, height_col, width_col, grad_im_);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in deformable_col2im: %s\n", cudaGetErrorString(err));
  }
}

template <typename scalar_t>
__global__ void deformable_col2im_coord_gpu_kernel(const int n, const scalar_t *data_col,
                                                   const scalar_t *data_im, const scalar_t *data_offset,
                                                   const int channels, const int height, const int width,
                                                   const int kernel_h, const int kernel_w,
                                                   const int pad_h, const int pad_w,
                                                   const int stride_h, const int stride_w,
                                                   const int dilation_h, const int dilation_w,
                                                   const int channel_per_deformable_group,
                                                   const int batch_size, const int offset_channels, const int deformable_group,
                                                   const int height_col, const int width_col, scalar_t *grad_offset)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    scalar_t val = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const scalar_t *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group *
                                                  batch_size * width_col * height_col;
    const scalar_t *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) *
                                                channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const scalar_t *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 *
                                                        kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const scalar_t offset_h = data_offset_ptr[data_offset_h_ptr];
      const scalar_t offset_w = data_offset_ptr[data_offset_w_ptr];
      scalar_t inv_h = h_in + i * dilation_h + offset_h;
      scalar_t inv_w = w_in + j * dilation_w + offset_w;
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
    const at::Tensor data_col, const at::Tensor data_im, const at::Tensor data_offset,
    const int channels, const int height, const int width, const int ksize_h,
    const int ksize_w, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    const int parallel_imgs, const int deformable_group, at::Tensor grad_offset)
{

  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  int num_kernels = height_col * width_col * 2 * ksize_h * ksize_w * deformable_group * parallel_imgs;
  int channel_per_deformable_group = channels * ksize_h * ksize_w / deformable_group;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_col.scalar_type(), "deformable_col2im_coord_gpu", ([&] {
        const scalar_t *data_col_ = data_col.data<scalar_t>();
        const scalar_t *data_im_ = data_im.data<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data<scalar_t>();
        scalar_t *grad_offset_ = grad_offset.data<scalar_t>();

        deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
            num_kernels, data_col_, data_im_, data_offset_, channels, height, width,
            ksize_h, ksize_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group,
            parallel_imgs, 2 * ksize_h * ksize_w * deformable_group, deformable_group,
            height_col, width_col, grad_offset_);
      }));
}








void shape_check(at::Tensor input, at::Tensor offset, at::Tensor *gradOutput,
                 at::Tensor weight, int kH, int kW, int dH, int dW, int padH,
                 int padW, int dilationH, int dilationW, int group,
                 int deformable_group) {
  AT_CHECK(weight.ndimension() == 4,
           "4D weight tensor (nOutputPlane,nInputPlane,kH,kW) expected, "
           "but got: %s",
           weight.ndimension());

  AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");

  AT_CHECK(kW > 0 && kH > 0,
           "kernel size should be greater than zero, but got kH: %d kW: %d", kH,
           kW);

  AT_CHECK((weight.size(2) == kH && weight.size(3) == kW),
           "kernel size should be consistent with weight, ",
           "but got kH: %d kW: %d weight.size(2): %d, weight.size(3): %d", kH,
           kW, weight.size(2), weight.size(3));

  AT_CHECK(dW > 0 && dH > 0,
           "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);

  AT_CHECK(
      dilationW > 0 && dilationH > 0,
      "dilation should be greater than 0, but got dilationH: %d dilationW: %d",
      dilationH, dilationW);

  int ndim = input.ndimension();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  AT_CHECK(ndim == 3 || ndim == 4, "3D or 4D input tensor expected but got: %s",
           ndim);

  long nInputPlane = weight.size(1) * group;
  long inputHeight = input.size(dimh);
  long inputWidth = input.size(dimw);
  long nOutputPlane = weight.size(0);
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;
  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;

  AT_CHECK(nInputPlane % deformable_group == 0,
           "input channels must divide deformable group size");

  if (outputWidth < 1 || outputHeight < 1)
    AT_ERROR(
        "Given input size: (%ld x %ld x %ld). "
        "Calculated output size: (%ld x %ld x %ld). Output size is too small",
        nInputPlane, inputHeight, inputWidth, nOutputPlane, outputHeight,
        outputWidth);

  AT_CHECK(input.size(1) == nInputPlane,
           "invalid number of input planes, expected: %d, but got: %d",
           nInputPlane, input.size(1));

  AT_CHECK((inputHeight >= kH && inputWidth >= kW),
           "input image is smaller than kernel");

  AT_CHECK((offset.size(2) == outputHeight && offset.size(3) == outputWidth),
           "invalid spatial size of offset, expected height: %d width: %d, but "
           "got height: %d width: %d",
           outputHeight, outputWidth, offset.size(2), offset.size(3));

  AT_CHECK((offset.size(1) == deformable_group * 2 * kH * kW),
           "invalid number of channels of offset");

  if (gradOutput != NULL) {
    AT_CHECK(gradOutput->size(dimf) == nOutputPlane,
             "invalid number of gradOutput planes, expected: %d, but got: %d",
             nOutputPlane, gradOutput->size(dimf));

    AT_CHECK((gradOutput->size(dimh) == outputHeight &&
              gradOutput->size(dimw) == outputWidth),
             "invalid size of gradOutput, expected height: %d width: %d , but "
             "got height: %d width: %d",
             outputHeight, outputWidth, gradOutput->size(dimh),
             gradOutput->size(dimw));
  }
}

int deform_conv_forward_cuda(at::Tensor input, at::Tensor weight,
                             at::Tensor offset, at::Tensor output,
                             at::Tensor columns, at::Tensor ones,
                             int dW, int dH, int padW, int padH,
                             int dilationW, int dilationH, int group,
                             int deformable_group, int im2col_step) {
  // todo: resize columns to include im2col: done
  // todo: add im2col_step as input
  // todo: add new output buffer and transpose it to output (or directly
  // transpose output) todo: possibly change data indexing because of
  // parallel_imgs

  int kH = weight.size(3);
  int kW = weight.size(2);

  shape_check(input, offset, NULL, weight, kH, kW, dH, dW, padH, padW,
              dilationH, dilationW, group, deformable_group);
  at::DeviceGuard guard(input.device());
  
  input = input.contiguous();
  offset = offset.contiguous();
  weight = weight.contiguous();

  // todo: assert batchsize dividable by im2col_step

  long batchSize = input.size(0);
  long nInputPlane = input.size(1);
  long inputHeight = input.size(2);
  long inputWidth = input.size(3);

  long nOutputPlane = weight.size(0);

  long outputWidth =
      (inputWidth + 2 * padW - (dilationW * (kW - 1) + 1)) / dW + 1;
  long outputHeight =
      (inputHeight + 2 * padH - (dilationH * (kH - 1) + 1)) / dH + 1;

  AT_CHECK((offset.size(0) == batchSize), "invalid batch size of offset");

  output = output.view({batchSize / im2col_step, im2col_step, nOutputPlane,
                        outputHeight, outputWidth});
  columns = at::zeros(
      {nInputPlane * kW * kH, im2col_step * outputHeight * outputWidth},
      input.options());

  if (ones.ndimension() != 2 ||
      ones.size(0) * ones.size(1) < outputHeight * outputWidth) {
    ones = at::ones({outputHeight, outputWidth}, input.options());
  }

  input = input.view({batchSize / im2col_step, im2col_step, nInputPlane,
                      inputHeight, inputWidth});
  offset =
      offset.view({batchSize / im2col_step, im2col_step,
                   deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  at::Tensor output_buffer =
      at::zeros({batchSize / im2col_step, nOutputPlane,
                 im2col_step * outputHeight, outputWidth},
                output.options());

  output_buffer = output_buffer.view(
      {output_buffer.size(0), group, output_buffer.size(1) / group,
       output_buffer.size(2), output_buffer.size(3)});

  for (int elt = 0; elt < batchSize / im2col_step; elt++) {
    deformable_im2col(input[elt], offset[elt], nInputPlane, inputHeight,
                      inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                      dilationW, im2col_step, deformable_group, columns);

    columns = columns.view({group, columns.size(0) / group, columns.size(1)});
    weight = weight.view({group, weight.size(0) / group, weight.size(1),
                          weight.size(2), weight.size(3)});

    // std::cout << "columns:\n";
    // std::cout << columns << "\n\n";

    for (int g = 0; g < group; g++) {
      output_buffer[elt][g] = output_buffer[elt][g]
                                  .flatten(1)
                                  .addmm_(weight[g].flatten(1), columns[g])
                                  .view_as(output_buffer[elt][g]);
    }
  }

  output_buffer = output_buffer.view(
      {output_buffer.size(0), output_buffer.size(1) * output_buffer.size(2),
       output_buffer.size(3), output_buffer.size(4)});

  output_buffer = output_buffer.view({batchSize / im2col_step, nOutputPlane,
                                      im2col_step, outputHeight, outputWidth});
  output_buffer.transpose_(1, 2);
  output.copy_(output_buffer);
  output = output.view({batchSize, nOutputPlane, outputHeight, outputWidth});

  // std::cout << "output:\n";
  // std::cout << output << "\n\n";

  input = input.view({batchSize, nInputPlane, inputHeight, inputWidth});
  offset = offset.view(
      {batchSize, deformable_group * 2 * kH * kW, outputHeight, outputWidth});

  return 1;
}






/*

template <typename T>
__global__ void DCNForward(
    const T* input,
    const T* offset,
    const T* weight,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int deformable_groups,
    const int im2col_step,
    T* output) {
  output[0] = input[0] * input[0];
}
*/
//*

at::Tensor DCN_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& weight,
    const int stride,
    const int padding,
    const int dilation,
    const int groups,
    const int deformable_groups,
    const int im2col_step) {
  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");

  auto batch_size = input.size(0);
  auto n_channels = weight.size(0);
  auto in_size = input.size(2);
  auto kernel_size = dilation * (weight.size(2) - 1) + 1;
  auto out_size = (in_size + (2 * padding) - kernel_size) / stride + 1;

  at::Tensor output = at::zeros({batch_size, n_channels, out_size, out_size}, input.options());

  at::Tensor buf0 = at::zeros({1}, input.options());
  at::Tensor buf1 = at::zeros({1}, input.options());

  int in_size0 = input.size(0);
  auto cur_im2col_step = std::min(in_size0, im2col_step);
  TORCH_CHECK(in_size0 % cur_im2col_step == 0);

  deform_conv_forward_cuda(
      input, weight, offset, output, buf0, buf1,
      stride, stride,
      padding, padding,
      dilation, dilation,
      groups, deformable_groups,
      cur_im2col_step);

  /*
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "DCN_forward", [&] {
    DCNForward<scalar_t><<<1, 1, 0, stream>>>(
        input.contiguous().data_ptr<scalar_t>(),
        weight.contiguous().data_ptr<scalar_t>(),
        offset.contiguous().data_ptr<scalar_t>(),
        buf0.contiguous().data_ptr<scalar_t>(),
        buf1.contiguous().data_ptr<scalar_t>(),
        stride,
        stride,
        padding,
        padding,
        dilation,
        dilation,
        groups,
        deformable_groups,
        im2col_step,
        output.data_ptr<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
  */
  return output;
}
// */



template <typename T>
__global__ void DCNBackward(
    const T* grad_output,
    const T* input,
    T* grad_input) {
  grad_input[0] = 2 * input[0] * grad_output[0];
}

at::Tensor DCN_backward_cuda(
    const at::Tensor& grad, const at::Tensor& input) {
  AT_ASSERTM(grad.device().is_cuda(), "grad must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(grad.device());

  at::Tensor grad_input =
      at::zeros({1}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "DCN_backward", [&] {
    DCNBackward<scalar_t><<<1, 1, 0, stream>>>(
        grad.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>());
  });
  return grad_input;
}


