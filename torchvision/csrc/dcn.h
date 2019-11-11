#pragma once

#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

at::Tensor DCN_forward(
    const Tensor& input,
    const Tensor& offset,
    const Tensor& weights,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    const int groups,
    const int deformable_groups,
    const int im2col_step    
    ) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return DCN_forward_cuda(input, offset, weights, stride_h, stride_w, pad_h, pad_w,
                      dilation_h, dilation_w, groups, deformable_groups, im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return DCN_forward_cpu(input, offset, weights, stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, groups, deformable_groups, im2col_step);
}

at::Tensor DCN_backward(const at::Tensor& grad, const at::Tensor& input) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return DCN_backward_cuda(grad, input);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return DCN_backward_cpu(grad, input);
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
      Variable input,
      Variable offset,
      Variable weights,
      int stride_h, int stride_w,
      int pad_h, int pad_w,
      int dilation_h, int dilation_w,
      const int groups,
      const int deformable_groups,
      const int im2col_step) {
    auto output = DCN_forward(input, offset, weights,
        stride.first, stride.second,
        pad.first, pad.second,
        dilation.first, dilation.second,
        groups, deformable_groups, im2col_step);
    ctx->save_for_backward({input, offset, weights});
    return {output,};
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto grad_in = DCN_backward(grad_output[0], input);
    return {grad_in, Variable(), Variable(),
            Variable(), Variable(), Variable(),
            Variable(), Variable(), Variable(),};
  }
};

Tensor dcn(
    const Tensor& input,
    const Tensor& offset,
    const Tensor& weights,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    const int groups,
    const int deformable_groups,
    const int im2col_step) {
  auto result = DCNFunction::apply(input, offset, weights, stride, pad,
                          dilation, groups, deformable_groups, im2col_step);
  return result[0];
}
