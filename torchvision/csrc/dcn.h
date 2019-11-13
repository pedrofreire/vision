#pragma once

#include "cpu/vision_cpu.h"

#ifdef WITH_CUDA
#include "cuda/vision_cuda.h"
#endif

at::Tensor DCN_forward(
    const Tensor& input,
    const Tensor& offset,
    const Tensor& weights,
    const std::pair<int, int>& stride,
    const std::pair<int, int>& pad,
    const std::pair<int, int>& dilation,
    const int groups,
    const int deformable_groups,
    const int im2col_step) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return DCN_forward_cuda(input, offset, weights, stride, pad,
                      dilation, groups, deformable_groups, im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return DCN_forward_cpu(input, offset, weights, stride, pad,
                    dilation, groups, deformable_groups, im2col_step);
}

std::tuple<at::Tensor> DCN_backward(
    const at::Tensor& grad,
    const Tensor& input,
    const Tensor& offset,
    const Tensor& weights,
    const std::pair<int, int>& stride,
    const std::pair<int, int>& pad,
    const std::pair<int, int>& dilation,
    const int groups,
    const int deformable_groups,
    const int im2col_step) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return DCN_backward_cuda(grad, input, offset, weights, stride, pad,
                      dilation, groups, deformable_groups, im2col_step);
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
      int64_t stride_h, int64_t stride_w,
      int64_t pad_h, int64_t pad_w,
      int64_t dilation_h, int64_t dilation_w,
      int64_t groups,
      int64_t deformable_groups,
      int64_t im2col_step) {
    auto output = DCN_forward(input, offset, weights,
        {stride_h, stride_w},
        {pad_h, pad_w},
        {dilation_h, dilation_w},
        groups, deformable_groups, im2col_step);
    ctx->save_for_backward({
        input, offset, weights,
        Variable(stride_h), Variable(stride_w),
        Variable(pad_h), Variable(pad_w),
        Variable(dilation_h), Variable(dilation_w),
        Variable(groups), Variable(deformable_groups), Variable(im2col_step)});
    return {output,};
  }

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_output) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto offset = saved[1];
    auto weight = saved[2];
    auto stride_h = saved[3];
    auto stride_w = saved[4];
    auto pad_h = saved[5];
    auto pad_w = saved[6];
    auto dilation_h = saved[7];
    auto dilation_w = saved[8];
    auto groups = saved[9];
    auto deformable_groups = saved[10];
    auto im2col_step = saved[11];

    auto grads = DCN_backward(grad_output[0],
        input, offset, weight,
        {stride_h, stride_w},
        {pad_h, pad_w},
        {dilation_h, dilation_w},
        groups, deformable_groups, im2col_step);
    auto grad_input = std::get<0>(grads);
    auto grad_offset = std::get<1>(grads);
    auto grad_weight = std::get<2>(grads);

    return {grad_input, grad_offset, grad_weight,
            Variable(), Variable(), Variable(),
            Variable(), Variable(), Variable(),
            Variable(), Variable(), Variable(),};
  }
};

Tensor dcn(
    const Tensor& input,
    const Tensor& offset,
    const Tensor& weights,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t groups,
    int64_t deformable_groups,
    int64_t im2col_step) {
  auto result = DCNFunction::apply(input, offset, weights, stride_h, stride_w, pad_h, pad_w,
                          dilation_h, dilation_w, groups, deformable_groups, im2col_step);
  return result[0];
}
