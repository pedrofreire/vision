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

    def _test_forward_cpu(self, x, rois, pool_h=5, pool_w=5):
        device = torch.device('cpu')
        y = self.fn(x, rois, pool_h, pool_w)
        gt_y = self.slow_fn(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y, y), 'RoIPool layer incorrect on CPU')

    def test_non_cont_grad(self):
        device = torch.device('cpu')
        n_channels = 25
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        grad_cont = torch.rand(3, 1, 5, 5, dtype=self.dtype, device=device)
        grad = grad_cont.permute(2, 1, 3, 0).contiguous().permute(3, 1, 0, 2)

        x1 = torch.rand(1, 1, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)

        pool_h, pool_w = 5, 5

        y1 = self.fn(x1, rois, pool_h, pool_w)
        y1.backward(grad_cont)

        y2 = self.fn(x2, rois, pool_h, pool_w)
        y2.backward(grad)

        self.assertTrue(torch.allclose(x1.grad, x2.grad), 'gradient incorrect')


    def test_roi_pool_gradcheck_cpu(self):
        device = torch.device('cpu')
        n_channels = 25
        x = torch.rand(1, n_channels, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        func = lambda z : self.fn(z, rois, 5, 5)

        self.assertTrue(gradcheck(func, (x,)), 'gradcheck failed for roi_pool CPU')
        self.assertTrue(gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for roi_pool CPU')

        self.assertTrue(gradcheck(self.get_script_fn(rois), (x,)), 'gradcheck failed for scripted roi_pool')



    def fn(*args, **kwargs):
        pass


    def slow_fn(*args, **kwargs):
        pass


class RoIPoolTester(RoIOpTester, unittest.TestCase):
    def slow_roi_pooling(self, x, rois, pool_h, pool_w, spatial_scale=1,
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

    def fn(self, x, rois, pool_h, pool_w):
        roi_pool = ops.RoIPool((pool_h, pool_w), 1)
        return roi_pool(x, rois)

    def slow_fn(self, *args, **kwargs):
        return self.slow_roi_pooling(*args, **kwargs)

    def get_script_fn(self, rois):
        @torch.jit.script
        def script_fn(input, rois):
            return ops.roi_pool(input, rois, 5, 1.0)[0]

        return lambda x: script_fn(x, rois)

    def test_forward_cpu(self):
        n_channels = 3
        self._test_forward_cpu(
                x=torch.rand(1, n_channels, 10, 10),
                rois=torch.tensor([[0, 0, 0, 4, 4]]))
        self._test_forward_cpu(
                x=torch.rand(1, n_channels, 10, 10),
                rois = torch.tensor(
                    [[0., 1., 0., 4., 0.],
                     [0., 2., 0., 3., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 2., 0., 2., 0.]],
                    dtype=self.dtype, device=device))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_pool_basic_cuda(self):
        device = torch.device('cuda')
        x = torch.rand(1, 1, 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (5, 5)
        roi_pool = ops.RoIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = self.slow_roi_pooling(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)

        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'RoIPool layer incorrect')

        y = roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device=device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'RoIPool layer incorrect')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_pool_cuda(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        x = torch.rand(2, 1, 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (5, 5)
        roi_pool = ops.RoIPool((pool_h, pool_w), 1)
        y = roi_pool(x, rois)

        gt_y = self.slow_roi_pooling(x, rois, pool_h, pool_w, device=device, dtype=self.dtype)

        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'RoIPool layer incorrect')

        y = roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device=device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'RoIPool layer incorrect')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_pool_gradcheck_cuda(self):
        device = torch.device('cuda')
        x = torch.rand(1, 1, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        m = ops.RoIPool((5, 5), 1).to(dtype=self.dtype, device=device)

        def func(input):
            return m(input, rois)

        self.assertTrue(gradcheck(func, (x,)), 'gradcheck failed for roi_pool CUDA')
        self.assertTrue(gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for roi_pool CUDA')

        @torch.jit.script
        def script_func(input, rois):
            return ops.roi_pool(input, rois, 5, 1.0)[0]

        self.assertTrue(gradcheck(lambda x: script_func(x, rois), (x,)),
                        'gradcheck failed for scripted roi_pool on CUDA')


class PSRoIPoolTester(RoIOpTester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float64


    def slow_ps_roi_pooling(self, x, rois, pool_h, pool_w, device, spatial_scale=1,
                            dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        num_input_channels = x.size(1)
        self.assertEqual(num_input_channels % (pool_h * pool_w), 0, "input channels must be divisible by ph * pw")
        num_output_channels = int(num_input_channels / (pool_h * pool_w))
        y = torch.zeros(rois.size(0), num_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        rois = torch.round(rois * spatial_scale).int()
        for n in range(0, x.size(0)):
            for r, roi in enumerate(rois):
                if roi[0] != n:
                    continue
                c_in = 0
                for c_out in range(0, num_output_channels):
                    roi_height = max(roi[4].item() - roi[2].item(), 1)
                    roi_width = max(roi[3].item() - roi[1].item(), 1)
                    bin_h, bin_w = roi_height / float(pool_h), roi_width / float(pool_w)

                    for j in range(0, pool_h):
                        start_h = int(np.floor(j * bin_h)) + roi[2].item()
                        end_h = int(np.ceil((j + 1) * bin_w)) + roi[2].item()

                        # range-check
                        start_h = min(max(start_h, 0), x.size(2))
                        end_h = min(max(end_h, 0), x.size(2))

                        for i in range(0, pool_w):
                            start_w = int(np.floor(i * bin_w)) + roi[1].item()
                            end_w = int(np.ceil((i + 1) * bin_w)) + roi[1].item()

                            # range-check
                            start_w = min(max(start_w, 0), x.size(3))
                            end_w = min(max(end_w, 0), x.size(3))

                            is_empty = (end_h <= start_h) or (end_w <= start_w)
                            area = (end_h - start_h) * (end_w - start_w)

                            if not is_empty:
                                t = torch.sum(x[n, c_in, slice(start_h, end_h), slice(start_w, end_w)])
                                y[r, c_out, j, i] = t / area
                            c_in += 1
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

    def test_forward_cpu(self):
        n_channels = 25
        self._test_forward_cpu(
                x=torch.rand(1, n_channels, 10, 10),
                rois=torch.tensor([[0, 0, 0, 4, 4]]))
        self._test_forward_cpu(
                x=torch.rand(1, n_channels, 10, 10),
                rois = torch.tensor(
                    [[0., 1., 0., 4., 0.],
                     [0., 2., 0., 3., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.],
                     [0., 2., 0., 2., 0.]],
                    dtype=self.dtype, device=device))


    def test_ps_roi_pool_basic_cpu(self):
        device = torch.device('cpu')
        pool_size = 3
        x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        ps_roi_pool = ops.PSRoIPool((pool_h, pool_w), 1)
        y = ps_roi_pool(x, rois)

        gt_y = self.slow_ps_roi_pooling(x, rois, pool_h, pool_w, device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y, y), 'PSRoIPool layer incorrect on CPU')

        y = ps_roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y, y), 'PSRoIPool layer incorrect on CPU')


    def test_ps_roi_pool_cpu(self):
        device = torch.device('cpu')
        pool_size = 5
        x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        ps_roi_pool = ops.PSRoIPool((pool_h, pool_w), 1)
        y = ps_roi_pool(x, rois)

        gt_y = self.slow_ps_roi_pooling(x, rois, pool_h, pool_w, device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y, y), 'PSRoIPool layer incorrect on CPU')

        y = ps_roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y, y), 'PSRoIPool layer incorrect on CPU')

    def test_ps_roi_pool_gradcheck_cpu(self):
        device = torch.device('cpu')
        pool_size = 5
        x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        m = ops.PSRoIPool((pool_size, pool_size), 1).to(dtype=self.dtype, device=device)

        def func(input):
            return m(input, rois)

        self.assertTrue(gradcheck(func, (x,)), 'gradcheck failed for PSRoIPool on CPU')
        self.assertTrue(gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for PSRoIPool on CPU')

        @torch.jit.script
        def script_fn(input, rois):
            return ops.ps_roi_pool(input, rois, 5, 1.0)[0]

        self.assertTrue(gradcheck(lambda z : script_fn(z, rois), (x,)),
                        'gradcheck failed for scripted ps_roi_pool on CPU')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_pool_basic_cuda(self):
        device = torch.device('cuda')
        pool_size = 3
        x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        ps_roi_pool = ops.PSRoIPool((pool_h, pool_w), 1)
        y = ps_roi_pool(x, rois)

        gt_y = self.slow_ps_roi_pooling(x, rois, pool_h, pool_w, device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'PSRoIPool layer incorrect')

        y = ps_roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'PSRoIPool layer incorrect')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_pool_cuda(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pool_size = 5
        x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        ps_roi_pool = ops.PSRoIPool((pool_h, pool_w), 1)
        y = ps_roi_pool(x, rois)

        gt_y = self.slow_ps_roi_pooling(x, rois, pool_h, pool_w, device, dtype=self.dtype)

        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'PSRoIPool layer incorrect')

        y = ps_roi_pool(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_pooling(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device, dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'PSRoIPool layer incorrect')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_pool_gradcheck_cuda(self):
        device = torch.device('cuda')
        pool_size = 5
        x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        m = ops.PSRoIPool((pool_size, pool_size), 1).to(dtype=self.dtype, device=device)

        def func(input):
            return m(input, rois)

        self.assertTrue(gradcheck(func, (x,)), 'gradcheck failed for PSRoIPool CUDA')
        self.assertTrue(gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for PSRoIPool CUDA')

        @torch.jit.script
        def script_func(input, rois):
            return ops.ps_roi_pool(input, rois, 5, 1.0)[0]

        self.assertTrue(gradcheck(lambda x: script_func(x, rois), (x,)),
                        'gradcheck failed for scripted ps_roi_pool on CUDA')


class RoIAlignTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float32
        torch.manual_seed(123)
        cls.x = torch.rand(1, 1, 10, 10, dtype=cls.dtype)
        cls.single_roi = torch.tensor([[0, 0, 0, 4, 4]],  # format is (xyxy)
                                      dtype=cls.dtype)
        cls.rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                                 [0, 0, 5, 4, 9],
                                 [0, 5, 5, 9, 9]],
                                dtype=cls.dtype)

        cls.gt_y_single = torch.tensor(
            [[[[0.41617328, 0.5040753, 0.25266218, 0.4296828, 0.29928464],
               [0.5210769, 0.57222337, 0.2524979, 0.32063985, 0.32635176],
               [0.73108256, 0.6114335, 0.62033176, 0.8188273, 0.5562218],
               [0.83115816, 0.70803946, 0.7084047, 0.74928707, 0.7769296],
               [0.54266506, 0.45964524, 0.5780159, 0.80522037, 0.7321807]]]], dtype=cls.dtype)

        cls.gt_y_multiple = torch.tensor(
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


    def test_roi_align_basic_cpu(self):
        device = torch.device('cpu')
        x = self.x.to(device)

        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)

        examples = [
                (self.single_roi, self.gt_y_single),
                (self.rois, self.gt_y_multiple)
                ]


        for rois, expected in examples:
            y = roi_align(x, rois)
            self.assertTrue(torch.allclose(expected, y), 'RoIAlign layer incorrect for single ROI on CPU')
            xtt = x.transpose(2, 3).contiguous().transpose(2, 3)
            y = roi_align(xtt, rois)
            self.assertTrue(torch.allclose(expected, y), 'RoIAlign layer incorrect for single ROI on CPU')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_align_basic_cuda(self):
        device = torch.device('cuda')
        x = self.x.to(device)
        single_roi = self.single_roi.to(device)
        gt_y_single = self.gt_y_single.to(device)

        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)
        y = roi_align(x, single_roi)

        self.assertTrue(torch.allclose(gt_y_single, y), 'RoIAlign layer incorrect for single ROI on CUDA')

        y = roi_align(x.transpose(2, 3).contiguous().transpose(2, 3), single_roi)
        self.assertTrue(torch.allclose(gt_y_single, y), 'RoIAlign layer incorrect for single ROI on CUDA')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_align_cuda(self):
        device = torch.device('cuda')
        x = self.x.to(device)
        rois = self.rois.to(device)
        gt_y_multiple = self.gt_y_multiple.to(device)

        pool_h, pool_w = (5, 5)
        roi_align = ops.RoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2).to(device=device)
        y = roi_align(x, rois)

        self.assertTrue(torch.allclose(gt_y_multiple, y), 'RoIAlign layer incorrect for multiple ROIs on CUDA')

        y = roi_align(x.transpose(2, 3).contiguous().transpose(2, 3), rois)
        self.assertTrue(torch.allclose(gt_y_multiple, y), 'RoIAlign layer incorrect for multiple ROIs on CUDA')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_roi_align_gradcheck_cuda(self):
        dtype = torch.float64
        device = torch.device('cuda')
        m = ops.RoIAlign((5, 5), 0.5, 1).to(dtype=dtype, device=device)
        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        rois = self.rois.to(device=device, dtype=dtype)

        def func(input):
            return m(input, rois)

        self.assertTrue(gradcheck(func, (x,)), 'gradcheck failed for RoIAlign CUDA')
        self.assertTrue(gradcheck(func, (x.transpose(2, 3),)), 'gradcheck failed for RoIAlign CUDA')

        @torch.jit.script
        def script_func(input, rois):
            return ops.roi_align(input, rois, 5, 0.5, 1)[0]

        self.assertTrue(gradcheck(lambda x: script_func(x, rois), (x,)),
                        'gradcheck failed for scripted roi_align on CUDA')

    def test_roi_align_gradcheck_cpu(self):
        dtype = torch.float64
        device = torch.device('cpu')
        m = ops.RoIAlign((5, 5), 0.5, 1).to(dtype=dtype, device=device)
        x = torch.rand(1, 1, 10, 10, dtype=dtype, device=device, requires_grad=True)
        rois = self.rois.to(device=device, dtype=dtype)

        fn = lambda z : m(z, rois)
        self.assertTrue(gradcheck(fn, (x,)), 'gradcheck failed for RoIAlign CPU')
        self.assertTrue(gradcheck(fn, (x.transpose(2, 3),)), 'gradcheck failed for RoIAlign CPU')

        @torch.jit.script
        def script_func(input, rois):
            return ops.roi_align(input, rois, 5, 0.5, 1)[0]

        self.assertTrue(gradcheck(lambda x: script_func(x, rois), (x,)), 'gradcheck failed for scripted roi_align')



def bilinear_interpolate(data, height, width, y, x):
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return 0.

    if y <= 0:
        y = 0.
    if x <= 0:
        x = 0.

    y_low, x_low = int(y), int(x)
    y_high, x_high = 0, 0

    if y_low >= height - 1:
        y_high = y_low = height - 1
        y = float(y_low)
    else:
        y_high = y_low + 1

    if x_low >= width - 1:
        x_high = x_low = width - 1
        x = float(x_low)
    else:
        x_high = x_low + 1

    ly = y - y_low
    lx = x - x_low
    hy, hx = 1. - ly, 1. - lx

    v1 = data[y_low * width + x_low]
    v2 = data[y_low * width + x_high]
    v3 = data[y_high * width + x_low]
    v4 = data[y_high * width + x_high]
    w1, w2, w3, w4 = hy * hx, hy * lx, ly * hx, ly * lx

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


class PSRoIAlignTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dtype = torch.float64

    def slow_ps_roi_align(self, in_data, rois, pool_h, pool_w, device, spatial_scale=1,
                          sampling_ratio=-1, dtype=torch.float64):
        if device is None:
            device = torch.device("cpu")
        num_input_channels = in_data.size(1)
        self.assertEqual(num_input_channels % (pool_h * pool_w), 0, "input channels must be divisible by ph * pw")
        num_output_channels = int(num_input_channels / (pool_h * pool_w))
        out_data = torch.zeros(rois.size(0), num_output_channels, pool_h, pool_w, dtype=dtype, device=device)

        for n in range(0, in_data.size(0)):
            for r, roi in enumerate(rois):
                if roi[0] != n:
                    continue
                roi[1:] = (roi[1:] * spatial_scale) - 0.5
                c_in = 0
                roi_height = float(roi[4].item() - roi[2].item())
                roi_width = float(roi[3].item() - roi[1].item())
                bin_h, bin_w = roi_height / float(pool_h), roi_width / float(pool_w)
                for c_out in range(0, num_output_channels):
                    for j in range(0, pool_h):
                        start_h = float(j) * bin_h + roi[2].item()

                        for i in range(0, pool_w):
                            start_w = float(i) * bin_w + roi[1].item()

                            roi_bin_grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(roi_height / pool_h))
                            roi_bin_grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(roi_width / pool_w))

                            val = 0.
                            for iy in range(0, roi_bin_grid_h):
                                y = start_h + (iy + 0.5) * bin_h / float(roi_bin_grid_h)
                                for ix in range(0, roi_bin_grid_w):
                                    x = start_w + (ix + 0.5) * bin_w / float(roi_bin_grid_w)
                                    val += bilinear_interpolate(
                                        in_data[n, c_in, :, :].flatten(),
                                        in_data.size(-2),
                                        in_data.size(-1),
                                        y, x
                                    )
                            count = roi_bin_grid_h * roi_bin_grid_w
                            out_data[r, c_out, j, i] = val / count
                            c_in += 1
        return out_data

    def _test_cpu(self, x, rois, pool_size, device):
        x = x.to(device)
        rois = rois.to(device)
        for z in [x, x.permute(0, 1, 3, 2)]:
            pool_h, pool_w = (pool_size, pool_size)
            ps_roi_align = ops.PSRoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2)
            y = ps_roi_align(z, rois)

            gt_y = self.slow_ps_roi_align(z, rois, pool_h, pool_w, device,
                                          spatial_scale=1, sampling_ratio=2,
                                          dtype=self.dtype)

            self.assertTrue(torch.allclose(gt_y, y), 'PSRoIAlign layer incorrect on CPU')
            self.assertTrue(gradcheck(lambda z : ps_roi_align(z, rois), (z,)), 'gradcheck failed for PSRoIAlign on CPU')

            @torch.jit.script 
            def script_fn(x, rois):
                return ops.roi_align(x, rois, 5, 0.5, 1)[0]

            self.assertTrue(gradcheck(lambda z : script_fn(z, rois), (x,)), 'gradcheck failed for PSRoIAlign on CPU')


    def test_ps_roi_align_basic_cpu(self):
        device = torch.device('cpu')
        pool_size = 5

        x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        self._test_cpu(x=x,
                  rois=rois,
                  pool_size=pool_size,
                  device=device
                )


    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_align_basic_cuda(self):
        device = torch.device('cuda')
        pool_size = 3
        x = torch.rand(1, 2 * (pool_size ** 2), 7, 7, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 5, 5]],  # format is (xyxy)
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        ps_roi_align = ops.PSRoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2)
        y = ps_roi_align(x, rois)

        gt_y = self.slow_ps_roi_align(x, rois, pool_h, pool_w, device,
                                      spatial_scale=1, sampling_ratio=2,
                                      dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'PSRoIAlign layer incorrect')

        y = ps_roi_align(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_align(x.permute(0, 1, 3, 2), rois, pool_h, pool_w, device,
                                      spatial_scale=1, sampling_ratio=-1,
                                      dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'PSRoIAlign layer incorrect')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_align_cuda(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pool_size = 5
        x = torch.rand(2, 2 * (pool_size ** 2), 10, 10, dtype=self.dtype, device=device)
        rois = torch.tensor([[0, 0, 0, 9, 9],  # format is (xyxy)
                             [0, 0, 5, 4, 9],
                             [0, 5, 5, 9, 9],
                             [1, 0, 0, 9, 9]],
                            dtype=self.dtype, device=device)

        pool_h, pool_w = (pool_size, pool_size)
        ps_roi_align = ops.PSRoIAlign((pool_h, pool_w), spatial_scale=1, sampling_ratio=2)
        y = ps_roi_align(x, rois)

        gt_y = self.slow_ps_roi_align(x, rois, pool_h, pool_w, device,
                                      spatial_scale=1, sampling_ratio=2,
                                      dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'PSRoIAlign layer incorrect')

        y = ps_roi_align(x.permute(0, 1, 3, 2), rois)
        gt_y = self.slow_ps_roi_align(x.permute(0, 1, 3, 2), rois, pool_h, pool_w,
                                      device, spatial_scale=1, sampling_ratio=2,
                                      dtype=self.dtype)
        self.assertTrue(torch.allclose(gt_y.cuda(), y), 'PSRoIAlign layer incorrect')

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA unavailable")
    def test_ps_roi_align_gradcheck_cuda(self):
        device = torch.device('cuda')
        pool_size = 5
        x = torch.rand(1, pool_size ** 2, 10, 10, dtype=self.dtype, device=device, requires_grad=True)
        rois = torch.tensor([
            [0, 0, 0, 9, 9],
            [0, 0, 5, 5, 9],
            [0, 5, 5, 9, 9]], dtype=self.dtype, device=device)

        m = ops.PSRoIAlign((pool_size, pool_size), spatial_scale=1,
                           sampling_ratio=2).to(dtype=self.dtype, device=device)

        def func(input):
            return m(input, rois)

        self.assertTrue(gradcheck(func, (x,)), 'gradcheck failed for PSRoIAlign CUDA')
        self.assertTrue(gradcheck(func, (x.permute(0, 1, 3, 2),)), 'gradcheck failed for PSRoIAlign CUDA')

        @torch.jit.script
        def script_func(input):
            return ops.ps_roi_align(input, rois, 5, 2.0, 1)[0]

        self.assertTrue(gradcheck(script_func, (x,)),
                        'gradcheck failed for scripted ps_roi_align on CUDA')


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
