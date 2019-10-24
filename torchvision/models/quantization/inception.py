import sys
import warnings
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception as inception_module
from torch import Tensor
from torch.jit.annotations import Optional
from torchvision.models.utils import load_state_dict_from_url


__all__ = [
    "QuantizableInception3",
    "inception_v3",
    "QuantizableInceptionOutputs",
]


model_urls = {
    # Inception v3 ported from TensorFlow
    "inception_v3_google": "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth"
}

QuantizableInceptionOutputs = namedtuple(
    "QuantizableInceptionOutputs", ["logits", "aux_logits"]
)
QuantizableInceptionOutputs.__annotations__ = {
    "logits": torch.Tensor,
    "aux_logits": Optional[torch.Tensor],
}


def inception_v3(pretrained_float_model=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained_float_model (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained_float_model:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "aux_logits" in kwargs:
            original_aux_logits = kwargs["aux_logits"]
            kwargs["aux_logits"] = True
        else:
            original_aux_logits = False
        model = QuantizableInception3(**kwargs)
        state_dict = load_state_dict_from_url(
            model_urls["inception_v3_google"], progress=progress
        )
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model

    return QuantizableInception3(**kwargs)


class QuantizableBasicConv2d(inception_module.BasicConv2d):
    def __init__(self, *args, **kwargs):
        super(QuantizableBasicConv2d, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ["conv", "bn", "relu"], inplace=True)


class QuantizableInceptionA(inception_module.InceptionA):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionA, self).__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionB(inception_module.InceptionB):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionB, self).__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionC(inception_module.InceptionC):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionC, self).__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionD(inception_module.InceptionD):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionD, self).__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionE(inception_module.InceptionE):
    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionE, self).__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.myop = nn.quantized.FloatFunctional()

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = self.myop.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = self.myop.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return self.myop.cat(outputs, 1)


class QuantizableInceptionAux(inception_module.InceptionAux):

    def __init__(self, *args, **kwargs):
        super(QuantizableInceptionAux, self).__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)

    def forward(self, x):
        x = self._forward(x)
        return x


class QuantizableInception3(inception_module.Inception3):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(QuantizableInception3, self).__init__(
            num_classes=num_classes,
            aux_logits=aux_logits,
            transform_input=transform_input,
            inception_blocks=[QuantizableBasicConv2d,
                                QuantizableInceptionA,
                                QuantizableInceptionB,
                                QuantizableInceptionC,
                                QuantizableInceptionD,
                                QuantizableInceptionE,
                                QuantizableInceptionAux]
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self._transform_input(x)
        x = self.quant(x)
        x, aux = self._forward(x)
        x = self.dequant(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted QuantizableInception3 always returns QuantizableInception3 Tuple")
            return QuantizableInceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in googlenet model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for m in self.modules():
            if type(m) == QuantizableBasicConv2d:
                torch.quantization.fuse_modules(m, ["conv", "bn", "relu"], inplace=True)
