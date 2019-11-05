#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <TH/TH.h>
#include <algorithm>

template <typename T>
void DCN_Forward(
    const T* input,
    T* output) {
  output[0] = 3;
}

template <typename T>
void DCNBackward(
    const T* grad,
    T* grad_input) {
  grad_input[0] = 5;
}

at::Tensor DCN_forward_cpu(const at::Tensor& input) {
  AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");

  at::Tensor output = at::zeros({1}, input.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "DCN_forward", [&] {
    DCN_Forward<scalar_t>(
        input.contiguous().data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>());
  });
  return output;
}

at::Tensor DCN_backward_cpu(
    const at::Tensor& grad) {
  // Check if input tensors are CPU tensors
  AT_ASSERTM(grad.device().is_cpu(), "grad must be a CPU tensor");

  at::Tensor grad_input = at::zeros({1}, grad.options());

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "DCN_backward", [&] {
    DCNBackward<scalar_t>(
        grad.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>());
  });
  return grad_input;
}
