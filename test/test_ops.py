from __future__ import division
import math
import numpy as np
import torch
from torch.autograd import gradcheck
from torch.nn.modules.utils import _pair

from torchvision import ops

from itertools import product
import unittest


class OpTester(object):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float64

    def test_forward_cpu_contiguous(self):
        self._test_forward(device=torch.device('cpu'), contiguous=True)

    def test_forward_cpu_non_contiguous(self):
        self._test_forward(device=torch.device('cpu'), contiguous=False)

    def test_backward_cpu_contiguous(self):
        self._test_backward(device=torch.device('cpu'), contiguous=True)

    def test_backward_cpu_non_contiguous(self):
        self._test_backward(device=torch.device('cpu'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_forward_cuda_contiguous(self):
        self._test_forward(device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_forward_cuda_non_contiguous(self):
        self._test_forward(device=torch.device('cuda'), contiguous=False)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_backward_cuda_contiguous(self):
        self._test_backward(device=torch.device('cuda'), contiguous=True)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_backward_cuda_non_contiguous(self):
        self._test_backward(device=torch.device('cuda'), contiguous=False)

    def _test_forward(self, device, contiguous):
        pass

    def _test_backward(self, device, contiguous):
        pass


class RoIOpTester(OpTester):
    def _test_forward(self, device, contiguous):
        pool_size = 5
        # n_channels % (pool_size ** 2) == 0 required for PS opeartions.
        n_channels = 2 * (pool_size ** 2)
        x = torch.rand(2, n_channels, 10, 10, dtype=self.dtype, device=device)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = pool_size, pool_size
        y = self.fn(x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1)
        gt_y = self.expected_fn(x, rois, pool_h, pool_w, spatial_scale=1,
                                sampling_ratio=-1, device=device, dtype=self.dtype)

        self.assertTrue(torch.allclose(gt_y, y))

    def _test_backward(self, device, contiguous):
        pool_size = 2
        x = torch.rand(1, 2 * (pool_size ** 2), 5, 5, dtype=self.dtype, device=device, requires_grad=True)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor([[0, 0, 0, 4, 4],  # format is (xyxy)
                             [0, 0, 2, 3, 4],
                             [0, 2, 2, 4, 4]],
                            dtype=self.dtype, device=device)

        def func(z):
            return self.fn(z, rois, pool_size, pool_size, spatial_scale=1, sampling_ratio=1)

        script_func = self.get_script_fn(rois, pool_size)

        self.assertTrue(gradcheck(func, (x,)))
        self.assertTrue(gradcheck(script_func, (x,)))

    def fn(*args, **kwargs):
        pass

    def get_script_fn(*args, **kwargs):
        pass

    def expected_fn(*args, **kwargs):
        pass


class RoIPoolTester(RoIOpTester, unittest.TestCase):
    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
        return ops.RoIPool((pool_h, pool_w), spatial_scale)(x, rois)

    def get_script_fn(self, rois, pool_size):
        @torch.jit.script
        def script_fn(input, rois, pool_size):
            # type: (torch.Tensor, torch.Tensor, int) -> torch.Tensor
            return ops.roi_pool(input, rois, pool_size, 1.0)[0]
        return lambda x: script_fn(x, rois, pool_size)

    def expected_fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1,
                    device=None, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")

        n_channels = x.size(1)
        y = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device)

        def get_slice(k, block):
            return slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block)))

        for roi_idx, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (int(round(x.item() * spatial_scale)) for x in roi[1:])
            roi_x = x[batch_idx, :, i_begin:i_end + 1, j_begin:j_end + 1]

            roi_h, roi_w = roi_x.shape[-2:]
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                for j in range(0, pool_w):
                    bin_x = roi_x[:, get_slice(i, bin_h), get_slice(j, bin_w)]
                    if bin_x.numel() > 0:
                        y[roi_idx, :, i, j] = bin_x.reshape(n_channels, -1).max(dim=1)[0]
        return y


class PSRoIPoolTester(RoIOpTester, unittest.TestCase):
    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
        return ops.PSRoIPool((pool_h, pool_w), 1)(x, rois)

    def get_script_fn(self, rois, pool_size):
        @torch.jit.script
        def script_fn(input, rois, pool_size):
            # type: (torch.Tensor, torch.Tensor, int) -> torch.Tensor
            return ops.ps_roi_pool(input, rois, pool_size, 1.0)[0]
        return lambda x: script_fn(x, rois, pool_size)

    def expected_fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1,
                    device=None, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        n_input_channels = x.size(1)
        self.assertEqual(n_input_channels % (pool_h * pool_w), 0, "input channels must be divisible by ph * pw")
        n_output_channels = int(n_input_channels / (pool_h * pool_w))
        y = torch.zeros(rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        def get_slice(k, block):
            return slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block)))

        for roi_idx, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (int(round(x.item() * spatial_scale)) for x in roi[1:])
            roi_x = x[batch_idx, :, i_begin:i_end + 1, j_begin:j_end + 1]

            roi_height = max(i_end - i_begin, 1)
            roi_width = max(j_end - j_begin, 1)
            bin_h, bin_w = roi_height / float(pool_h), roi_width / float(pool_w)

            for i in range(0, pool_h):
                for j in range(0, pool_w):
                    bin_x = roi_x[:, get_slice(i, bin_h), get_slice(j, bin_w)]
                    if bin_x.numel() > 0:
                        area = bin_x.size(-2) * bin_x.size(-1)
                        for c_out in range(0, n_output_channels):
                            c_in = c_out * (pool_h * pool_w) + pool_w * i + j
                            t = torch.sum(bin_x[c_in, :, :])
                            y[roi_idx, c_out, i, j] = t / area
        return y


def bilinear_interpolate(data, y, x, snap_border=False):
    height, width = data.shape

    if snap_border:
        if -1 < y <= 0:
            y = 0
        elif height - 1 <= y < height:
            y = height - 1

        if -1 < x <= 0:
            x = 0
        elif width - 1 <= x < width:
            x = width - 1

    y_low = int(math.floor(y))
    x_low = int(math.floor(x))
    y_high = y_low + 1
    x_high = x_low + 1

    wy_h = y - y_low
    wx_h = x - x_low
    wy_l = 1 - wy_h
    wx_l = 1 - wx_h

    val = 0
    for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
        for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
            if 0 <= yp < height and 0 <= xp < width:
                val += wx * wy * data[yp, xp]
    return val


class RoIAlignTester(RoIOpTester, unittest.TestCase):
    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
        return ops.RoIAlign((pool_h, pool_w), spatial_scale=spatial_scale,
                            sampling_ratio=sampling_ratio)(x, rois)

    def get_script_fn(self, rois, pool_size):
        @torch.jit.script
        def script_fn(input, rois, pool_size):
            # type: (torch.Tensor, torch.Tensor, int) -> torch.Tensor
            return ops.roi_align(input, rois, pool_size, 1.0)[0]
        return lambda x: script_fn(x, rois, pool_size)

    def expected_fn(self, in_data, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1,
                    device=None, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        n_channels = in_data.size(1)
        out_data = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device)

        for r, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (x.item() * spatial_scale for x in roi[1:])

            roi_h = i_end - i_begin
            roi_w = j_end - j_begin
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                start_h = i_begin + i * bin_h
                grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
                for j in range(0, pool_w):
                    start_w = j_begin + j * bin_w
                    grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))

                    for channel in range(0, n_channels):

                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(in_data[batch_idx, channel, :, :], y, x, snap_border=True)
                        val /= grid_h * grid_w

                        out_data[r, channel, i, j] = val
        return out_data


class PSRoIAlignTester(RoIOpTester, unittest.TestCase):
    def fn(self, x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, **kwargs):
        return ops.PSRoIAlign((pool_h, pool_w), spatial_scale=spatial_scale,
                              sampling_ratio=sampling_ratio)(x, rois)

    def get_script_fn(self, rois, pool_size):
        @torch.jit.script
        def script_fn(input, rois, pool_size):
            # type: (torch.Tensor, torch.Tensor, int) -> torch.Tensor
            return ops.ps_roi_align(input, rois, pool_size, 1.0)[0]
        return lambda x: script_fn(x, rois, pool_size)

    def expected_fn(self, in_data, rois, pool_h, pool_w, device, spatial_scale=1,
                    sampling_ratio=-1, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        n_input_channels = in_data.size(1)
        self.assertEqual(n_input_channels % (pool_h * pool_w), 0, "input channels must be divisible by ph * pw")
        n_output_channels = int(n_input_channels / (pool_h * pool_w))
        out_data = torch.zeros(rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        for r, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (x.item() * spatial_scale - 0.5 for x in roi[1:])

            roi_h = i_end - i_begin
            roi_w = j_end - j_begin
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                start_h = i_begin + i * bin_h
                grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
                for j in range(0, pool_w):
                    start_w = j_begin + j * bin_w
                    grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))
                    for c_out in range(0, n_output_channels):
                        c_in = c_out * (pool_h * pool_w) + pool_w * i + j

                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(in_data[batch_idx, c_in, :, :], y, x, snap_border=True)
                        val /= grid_h * grid_w

                        out_data[r, c_out, i, j] = val
        return out_data


class NMSTester(unittest.TestCase):
    def reference_nms(self, boxes, scores, iou_threshold):
        """
        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
        Returns:
             picked: a list of indexes of the kept boxes
        """
        picked = []
        _, indexes = scores.sort(descending=True)
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:]
            rest_boxes = boxes[indexes, :]
            iou = ops.box_iou(rest_boxes, current_box.unsqueeze(0)).squeeze(1)
            indexes = indexes[iou <= iou_threshold]

        return torch.as_tensor(picked)

    def _create_tensors_with_iou(self, N, iou_thresh):
        # force last box to have a pre-defined iou with the first box
        # let b0 be [x0, y0, x1, y1], and b1 be [x0, y0, x1 + d, y1],
        # then, in order to satisfy ops.iou(b0, b1) == iou_thresh,
        # we need to have d = (x1 - x0) * (1 - iou_thresh) / iou_thresh
        boxes = torch.rand(N, 4) * 100
        boxes[:, 2:] += boxes[:, :2]
        boxes[-1, :] = boxes[0, :]
        x0, y0, x1, y1 = boxes[-1].tolist()
        boxes[-1, 2] += (x1 - x0) * (1 - iou_thresh) / iou_thresh
        scores = torch.rand(N)
        return boxes, scores

    def test_nms(self):
        err_msg = 'NMS incompatible between CPU and reference implementation for IoU={}'
        for iou in [0.2, 0.5, 0.8]:
            boxes, scores = self._create_tensors_with_iou(1000, iou)
            keep_ref = self.reference_nms(boxes, scores, iou)
            keep = ops.nms(boxes, scores, iou)
            self.assertTrue(torch.allclose(keep, keep_ref), err_msg.format(iou))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_nms_cuda(self):
        err_msg = 'NMS incompatible between CPU and CUDA for IoU={}'

        for iou in [0.2, 0.5, 0.8]:
            boxes, scores = self._create_tensors_with_iou(1000, iou)
            r_cpu = ops.nms(boxes, scores, iou)
            r_cuda = ops.nms(boxes.cuda(), scores.cuda(), iou)

            self.assertTrue(torch.allclose(r_cpu, r_cuda.cpu()), err_msg.format(iou))


class DeformConvTester(OpTester, unittest.TestCase):
    def expected_fn(self, x, offsets, weights, *args, stride=1, pad=0, dilation=1):
        stride_h, stride_w = _pair(stride)
        pad_h, pad_w = _pair(pad)
        dil_h, dil_w = _pair(dilation)
        weights_h, weights_w = weights.shape[-2:]

        n_batches, n_in_channels, in_h, in_w = x.shape
        n_out_channels = weights.shape[0]

        out_h = (in_h + 2 * pad_h - (dil_h * (weights_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weights_w - 1) + 1)) // stride_w + 1

        n_offset_grps = offsets.shape[1] // (2 * weights_h * weights_w)
        in_c_per_offset_grp = n_in_channels // n_offset_grps

        n_weight_grps = n_in_channels // weights.shape[1]
        in_c_per_weight_grp = weights.shape[1]
        out_c_per_weight_grp = weights.shape[0] // n_out_channels

        out = torch.zeros(n_batches, n_out_channels, out_h, out_w, device=x.device, dtype=x.dtype)
        for b in range(n_batches):
            for c_out in range(n_out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        for di in range(weights_h):
                            for dj in range(weights_w):
                                for c in range(in_c_per_weight_grp):
                                    weight_grp = c_out // out_c_per_weight_grp
                                    c_in = weight_grp * in_c_per_weight_grp + c

                                    offset_grp = c_in // in_c_per_offset_grp
                                    offset_idx = 2 * (offset_grp * (weights_h * weights_w) + di * weights_w + dj)

                                    pi = stride_h * i - pad_h + dil_h * di + offsets[b, offset_idx, i, j]
                                    pj = stride_w * j - pad_w + dil_w * dj + offsets[b, offset_idx + 1, i, j]

                                    out[b, c_out, i, j] += weights[c_out, c, di, dj] * bilinear_interpolate(x[b, c_in, :, :], pi, pj)
        return out

    def _test_forward(self, device, contiguous):
        n_weight_grps = 2
        n_offset_grps = 3

        stride = (2, 1)
        pad = (1, 0)
        dilation = (2, 1)

        stride_h, stride_w = stride
        pad_h, pad_w = pad
        dil_h, dil_w = dilation
        weight_h, weight_w = (3, 2)
        in_h, in_w = (5, 4)

        out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) // stride_h + 1
        out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) // stride_w + 1

        x = torch.rand(1, 6, in_h, in_w, device=device, dtype=self.dtype)
        offset = torch.randn(1, n_offset_grps * 8, out_h, out_w, device=device, dtype=self.dtype)
        weight = torch.randn(2, 6 // n_weight_grps, 2, 2, device=device, dtype=self.dtype)

        if not contiguous:
            x = x.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
            offset = offset.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            weight = weight.permute(3, 2, 0, 1).contiguous().permute(2, 3, 1, 0)

        res = ops.deform_conv(x, offset, weight, stride=stride, pad=pad, dilation=dilation)
        expected = self.expected_fn(x, offset, weight, stride=stride, pad=pad, dilation=dilation)

        self.assertTrue(torch.allclose(res, expected), '\nx:\n{}\nres:\n{}\nexp:\n{}'.format(x, res, expected))

    def _test_backward(self, device, contiguous):
        n_weight_grps = 2
        n_offset_grps = 3
        x = torch.rand(1, 6, 5, 4, requires_grad=True, device=device, dtype=self.dtype)
        offset = torch.randn(1, n_offset_grps * 8, 4, 3, device=device, dtype=self.dtype)
        weight = torch.randn(2, 6 // n_weight_grps, 2, 2, device=device, dtype=self.dtype)

        if not contiguous:
            x = x.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
            offset = offset.permute(1, 3, 0, 2).contiguous().permute(2, 0, 3, 1)
            weight = weight.permute(3, 2, 0, 1).contiguous().permute(2, 3, 1, 0)

        def fn(z):
            return ops.deform_conv(z, offset, weight)

        gradcheck(fn, (x,), nondet_tol=1e-5)


if __name__ == '__main__':
    unittest.main()
