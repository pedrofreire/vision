import torch
from torch import nn, Tensor

from torch.nn.modules.utils import _pair
from torch.jit.annotations import List


def deform_conv(input, offset, weight, *args, stride=1, pad=0, dilation=1, im2col_step=64):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    Performs Deformable Convolution described in Deformable Convolution Networks

    Arguments:
        input (Tensor[N, C, H, W]): input tensor
        offset (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from. If a single Tensor is passed,
            then the first column should contain the batch index. If a list of Tensors
            is passed, then each Tensor will correspond to the boxes for an element i
            in a batch
        output_size (int or Tuple[int, int]): the size of the output after the cropping
            is performed, as (height, width)
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0

    Returns:
        output (Tensor[K, C, output_size[0], output_size[1]])
    """

    stride = _pair(stride)
    pad = _pair(pad)
    dilation = _pair(dilation)

    stride_h, stride_w = stride
    pad_h, pad_w = pad
    dil_h, dil_w = dilation
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, in_h, in_w = input.shape

    kernel_h = (weights_h - 1) * dil_h + 1
    kernel_w = (weights_w - 1) * dil_w + 1

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    output = torch.ops.torchvision.deform_conv(
                input,
                offset,
                weight,
                *stride,
                *pad,
                *dilation,
                n_weight_grps,
                n_offset_grps,
                im2col_step)
    return output


class DeformConv(nn.Module):
    """
    See deform_conv
    """
    def __init__(self, output_size, spatial_scale):
        super(DeformConv, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input):
        return deform_conv(input)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ')'
        return tmpstr
