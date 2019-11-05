#pragma once

#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

at::Tensor DCN_forward(
    const at::Tensor& input) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return DCN_forward_cuda(input);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return DCN_forward_cpu(input);
}

at::Tensor DCN_backward(const at::Tensor& grad) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return DCN_backward_cuda(grad);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return DCN_backward_cpu(grad);
}

using namespace at;
using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class DCNFunction : public torch::autograd::Function<DCNFunction> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      Variable input) {
    ctx->saved_data["important_data"] = {42};
    auto output = DCN_forward(input);
    return {output, };
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    // Use data saved in forward
    auto important = ctx->saved_data["input_shape"].toIntList();
    auto grad_in = DCN_backward(grad_output[0]);
    return {grad_in};
  }
};

Tensor dcn(
    const Tensor& input) {
  auto result = DCNFunction::apply(input);
  return result[0];
}
