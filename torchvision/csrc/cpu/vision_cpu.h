#pragma once
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width);

at::Tensor ROIPool_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width);

at::Tensor ROIAlign_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor ROIAlign_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio);

std::tuple<at::Tensor, at::Tensor> PSROIPool_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width);

at::Tensor PSROIPool_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& mapping_channel,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width);

std::tuple<at::Tensor, at::Tensor> PSROIAlign_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio);

at::Tensor PSROIAlign_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& mapping_channel,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const int batch_size,
    const int channels,
    const int height,
    const int width);

at::Tensor nms_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const float iou_threshold);

at::Tensor DCN_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& offset,
    const at::Tensor& weights,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int deformable_groups,
    int im2col_step);

at::Tensor DCN_backward_cpu(
    const at::Tensor& grad, const at::Tensor& input);

