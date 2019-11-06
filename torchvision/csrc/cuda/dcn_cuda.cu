#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include "cuda_helpers.h"

template <typename T>
__global__ void DCNForward(
    const T* input,
    T* output) {
  output[0] = 10;
}

template <typename T>
__global__ void DCNBackward(
    const T* grad_output,
    T* grad_input) {
  grad_input[0] = 20;
}

at::Tensor DCN_forward_cuda(
    const at::Tensor& input) {
  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");

  at::cuda::CUDAGuard device_guard(input.device());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  at::Tensor output = at::zeros(
      {1}, input.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "DCN_forward", [&] {
    DCNForward<scalar_t><<<1, 1, 0, stream>>>(
        input.contiguous().data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>());
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

at::Tensor DCN_backward_cuda(
    const at::Tensor& grad) {
  AT_ASSERTM(grad.device().is_cuda(), "grad must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(grad.device());

  at::Tensor grad_input =
      at::zeros({1}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "DCN_backward", [&] {
    DCNBackward<scalar_t><<<1, 1, 0, stream>>>(
        grad.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>());
  });
  return grad_input;
}
