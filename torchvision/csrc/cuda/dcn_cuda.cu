#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include "cuda_helpers.h"

void deform_conv_forward_cuda(at::Tensor input, at::Tensor weight,
                             at::Tensor offset, at::Tensor output,
                             at::Tensor columns, at::Tensor ones,
                             int dW, int dH, int padW, int padH,
                             int dilationW, int dilationH, int group,
                             int deformable_group, int im2col_step) {
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

  at::cuda::CUDAGuard device_guard(input.device());

  auto batch_size = input.size(0);
  auto n_channels = weight.size(0);
  auto in_size = input.size(2);
  auto kernel_size = dilation * (weight.size(2) - 1) - 1;
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


