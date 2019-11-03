from __future__ import division
import numpy as np
import torch
from torch.autograd import gradcheck

from torchvision import ops

from itertools import product
import unittest

from collections import namedtuple
Example = namedtuple('Example', ['x', 'rois'])

class RoIOpTester:
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


    def fn(*args, **kwargs):
        pass


    def slow_fn(*args, **kwargs):
        pass


class RoIPoolTester(RoIOpTester, unittest.TestCase):
    def roi_pool_reference(self, x, rois, pool_h, pool_w, spatial_scale=1,
                         device=None, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")

        n_channels = x.size(1)
        y = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device)

        rois = torch.round(rois * spatial_scale)

        get_slice = (lambda k, block :
                    slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block))))

        for roi_idx, roi in enumerate(rois):
            batch_idx, j_begin, i_begin, j_end, i_end = (int(x) for x in roi)
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

    def _test_forward(self, *, device, contiguous):
        x=torch.rand(1, 3, 10, 10, dtype=self.dtype, device=device)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)

        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        pool_h, pool_w = 5, 5
        roi_pool = ops.RoIPool((pool_h, pool_w), 1)
        y =  roi_pool(x, rois)
        gt_y = self.roi_pool_reference(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y, y))


    def _test_backward(self, *, device, contiguous):
        x = torch.rand(1, 3, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)

        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        roi_pool = ops.RoIPool((5, 5), 1).to(dtype=self.dtype, device=device)
        fn = lambda z : roi_pool(z, rois)

        @torch.jit.script
        def script_func(input, rois):
            return ops.roi_pool(input, rois, 5, 1.0)[0]
        script_fn = lambda z : script_func(z, rois)

        self.assertTrue(gradcheck(fn, (x,)))
        self.assertTrue(gradcheck(script_fn, (x,)))


class PSRoIPoolTester(RoIOpTester, unittest.TestCase):

    def ps_roi_pool_reference(self, x, rois, pool_h, pool_w, device, spatial_scale=1,
                            dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        n_input_channels = x.size(1)
        self.assertEqual(n_input_channels % (pool_h * pool_w), 0, "input channels must be divisible by ph * pw")
        n_output_channels = int(n_input_channels / (pool_h * pool_w))
        y = torch.zeros(rois.size(0), n_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        rois = torch.round(rois * spatial_scale).int()

        get_slice = (lambda k, block :
                    slice(int(np.floor(k * block)), int(np.ceil((k + 1) * block))))

        for roi_idx, roi in enumerate(rois):
            batch_idx, j_begin, i_begin, j_end, i_end = (int(x) for x in roi)
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
                            c_in = c_out * (pool_h * pool_w) + pool_w*i + j
                            t = torch.sum(bin_x[c_in, :, :])
                            y[roi_idx, c_out, i, j] = t / area
        return y
    

    def fn(self, x, rois, pool_h, pool_w):
        ps_roi_pool = ops.PSRoIPool((pool_h, pool_w), 1)
        return ps_roi_pool(x, rois)

    def slow_fn(self, *args, **kwargs):
        return self.slow_ps_roi_pooling(*args, **kwargs)

    def get_script_fn(self, rois):
        @torch.jit.script
        def script_fn(input, rois):
            return ops.ps_roi_pool(input, rois, 5, 1.0)[0]

        return lambda x: script_fn(x, rois)


    def _test_forward(self, *, device, contiguous):
        pool_size = 5
        x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = pool_size, pool_size
        ps_roi_pool = ops.PSRoIPool((pool_h, pool_w), 1)
        y =  ps_roi_pool(x, rois)
        gt_y = self.ps_roi_pool_reference(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y, y))


    def _test_backward(self, *, device, contiguous):
        pool_size = 5
        x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        ps_roi_pool = ops.PSRoIPool((5, 5), 1).to(dtype=self.dtype, device=device)
        fn = lambda z : ps_roi_pool(z, rois)

        @torch.jit.script
        def script_func(input, rois):
            return ops.ps_roi_pool(input, rois, 5, 1.0)[0]
        script_fn = lambda z : script_func(z, rois)

        self.assertTrue(gradcheck(fn, (x,)))
        self.assertTrue(gradcheck(script_fn, (x,)))



class RoIAlignTester(RoIOpTester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float32
        torch.manual_seed(123)
        cls.x = torch.rand(1, 1, 10, 10, dtype=cls.dtype)
        cls.rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                                 [0, 0, 5, 4, 9],
                                 [0, 5, 5, 9, 9]],
                                dtype=cls.dtype)
        cls.expected = torch.tensor(
            [[[[0.49311584, 0.35972416, 0.40843594, 0.3638034, 0.49751836],
               [0.70881474, 0.75481665, 0.5826779, 0.34767765, 0.46865487],
               [0.4740328, 0.69306874, 0.3617804, 0.47145438, 0.66130304],
               [0.6861706, 0.17634538, 0.47194335, 0.42473823, 0.37930614],
               [0.62666404, 0.49973848, 0.37911576, 0.5842756, 0.7176864]]],
             [[[0.67499936, 0.6607055, 0.42656037, 0.46134934, 0.42144877],
               [0.7471722, 0.7235433, 0.14512213, 0.13031253, 0.289369],
               [0.8443615, 0.6659734, 0.23614208, 0.14719573, 0.4268827],
               [0.69429564, 0.5621515, 0.5019923, 0.40678093, 0.34556213],
               [0.51315194, 0.7177093, 0.6494485, 0.6775592, 0.43865064]]],
             [[[0.24465509, 0.36108392, 0.64635646, 0.4051828, 0.33956185],
               [0.49006107, 0.42982674, 0.34184104, 0.15493104, 0.49633422],
               [0.54400194, 0.5265246, 0.22381854, 0.3929715, 0.6757667],
               [0.32961223, 0.38482672, 0.68877804, 0.71822757, 0.711909],
               [0.561259, 0.71047884, 0.84651315, 0.8541089, 0.644432]]]], dtype=cls.dtype)


    def _test_forward(self, *, device, contiguous):
        x = self.x.to(device)
        if not contiguous:
            x = x.transpose(2, 3).contiguous().transpose(2, 3)
        rois = self.rois.to(device)
        expected = self.expected.to(device)

        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)
        y = roi_align(x, self.rois)
        self.assertTrue(torch.allclose(expected, y))


    def _test_backward(self, *, device, contiguous):
        dtype = torch.float64

        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)

        rois = self.rois.to(dtype=dtype, device=device)

        roi_align = ops.RoIAlign((5, 5), 0.5, 1).to(dtype=dtype, device=device)
        fn = lambda z : roi_align(z, rois)

        @torch.jit.script
        def script_func(input, rois):
            return ops.roi_align(input, rois, 5, 0.5, 1)[0]
        script_fn = lambda z : script_func(z, rois)

        self.assertTrue(gradcheck(fn, (x,)))
        self.assertTrue(gradcheck(script_fn, (x,)))


def bilinear_interpolate(data, height, width, y, x):
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return 0.

    y = max(0, y)
    x = max(0, x)

    if int(y) < height - 1:
        y_low = int(y)
        y_high = y_low + 1
    else:
        y = height - 1
        y_low = y
        y_high = y

    if int(x) < width - 1:
        x_low = int(x)
        x_high = x_low + 1
    else:
        x = width - 1
        x_low = x
        x_high = x

    wy_h = y - y_low
    wy_l = 1 - wy_h

    wx_h = x - x_low
    wx_l = 1 - wx_h

    val = 0
    for wx, x in zip((wx_l, wx_h), (x_low, x_high)):
        for wy, y in zip((wy_l, wy_h), (y_low, y_high)):
            val += wx * wy * data[y * width + x]
    return val


class PSRoIAlignTester(RoIOpTester, unittest.TestCase):
    def ps_roi_align_reference(self, in_data, rois, pool_h, pool_w, device, spatial_scale=1,
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
                        c_in = c_out * (pool_h * pool_w) + pool_w*i + j

                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(
                                    in_data[batch_idx, c_in, :, :].flatten(),
                                    in_data.size(-2),
                                    in_data.size(-1),
                                    y, x
                                )
                        val /= grid_h * grid_w

                        out_data[r, c_out, i, j] = val
        return out_data


    def _test_forward(self, *, device, contiguous):
        pool_size = 5
        x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = pool_size, pool_size
        ps_roi_align = ops.PSRoIAlign((pool_h, pool_w), 1, -1)
        y = ps_roi_align(x, rois)
        gt_y = self.ps_roi_align_reference(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y, y), 'RoIAlign layer incorrect on CPU')


    def _test_backward(self, *, device, contiguous):
        pool_size = 5
        x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        if not contiguous:
            x = x.permute(0, 1, 3, 2)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        ps_roi_align = ops.PSRoIAlign((5, 5), 1, 1).to(dtype=self.dtype, device=device)
        fn = lambda z : ps_roi_align(z, rois)

        @torch.jit.script
        def script_func(input, rois):
            return ops.ps_roi_align(input, rois, 5, 1.0)[0]
        script_fn = lambda z : script_func(z, rois)

        self.assertTrue(gradcheck(fn, (x,)), 'gradcheck failed for roi_align CPU')
        self.assertTrue(gradcheck(script_fn, (x,)), 'gradcheck failed for scripted roi_align')


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

    def _create_tensors(self, N):
        boxes = torch.rand(N, 4) * 100
        boxes[:, 2:] += torch.rand(N, 2) * 100
        scores = torch.rand(N)
        return boxes, scores

    def test_nms(self):
        boxes, scores = self._create_tensors(1000)
        err_msg = 'NMS incompatible between CPU and reference implementation for IoU={}'
        for iou in [0.2, 0.5, 0.8]:
            keep_ref = self.reference_nms(boxes, scores, iou)
            keep = ops.nms(boxes, scores, iou)
            self.assertTrue(torch.allclose(keep, keep_ref), err_msg.format(iou))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_nms_cuda(self):
        boxes, scores = self._create_tensors(1000)
        err_msg = 'NMS incompatible between CPU and CUDA for IoU={}'

        for iou in [0.2, 0.5, 0.8]:
            r_cpu = ops.nms(boxes, scores, iou)
            r_cuda = ops.nms(boxes.cuda(), scores.cuda(), iou)

            self.assertTrue(torch.allclose(r_cpu, r_cuda.cpu()), err_msg.format(iou))


if __name__ == '__main__':
    unittest.main()
