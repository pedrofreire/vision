#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <TH/TH.h>
#include <algorithm>

template <typename T>
void DCN_Forward(
    const T* input,
    T* output) {
  output[0] = input[0] * input[0];
}

at::Tensor DCN_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& weights,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int group,
    int deformable_group,
    int im2col_block) {
  AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");

  at::Tensor output = at::zeros({1}, input.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "DCN_forward", [&] {
    DCN_Forward<scalar_t>(
        input.contiguous().data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>());
  });
  return output;
}

template <typename T>
void DCN_Backward(
    const T* grad, const T* input,
    T* grad_input) {
  grad_input[0] = 2 * input[0] * grad[0];
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> DCN_backward_cpu(
    const at::Tensor& grad, const at::Tensor& input) {
  // Check if input tensors are CPU tensors
  AT_ASSERTM(grad.device().is_cpu(), "grad must be a CPU tensor");

  at::Tensor grad_input = at::zeros({1}, grad.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "DCN_backward", [&] {
    DCN_Backward<scalar_t>(
        grad.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>());
  });

  return {grad_input, grad_input, grad_input};
}
