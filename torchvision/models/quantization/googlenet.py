import warnings
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torchvision.models.googlenet
import sys
from torch import Tensor
from torch.jit.annotations import Optional

googlenet_module = sys.modules['torchvision.models.googlenet']

__all__ = ['QuantizableGoogLeNet', 'googlenet', "QuantizableGoogLeNetOutputs", "_QuantizableGoogLeNetOutputs"]

model_urls = {
    # GoogLeNet ported from TensorFlow
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

QuantizableGoogLeNetOutputs = namedtuple('QuantizableGoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
QuantizableGoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

# Script annotations failed with _QuantizableGoogleNetOutputs = namedtuple ...
# _QuantizableGoogLeNetOutputs set here for backwards compat
_QuantizableGoogLeNetOutputs = QuantizableGoogLeNetOutputs


def googlenet(pretrained_float_model=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.

    Args:
        pretrained_float_model (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained_float_model is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained_float_model:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained_float_model googlenet model are NOT pretrained_float_model, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = QuantizableGoogLeNet(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1, model.aux2
        return model

    return QuantizableGoogLeNet(**kwargs)


class QuantizableBasicConv2d(googlenet_module.BasicConv2d):

    def __init__(self, *args, **kwargs):
        super(QuantizableBasicConv2d, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class QuantizableInception(googlenet_module.Inception):

    def __init__(self, *args, **kwargs):
        super(QuantizableInception, self).__init__(basic_conv2d=QuantizableBasicConv2d, *args, **kwargs)
        self.cat = nn.quantized.FloatFunctional()

    def forward(self, x):
        outputs = self._forward(x)
        return self.cat.cat(outputs, 1)


class QuantizableGoogLeNet(googlenet_module.GoogLeNet):

    def __init__(self, *args, **kwargs):
        super(QuantizableGoogLeNet, self).__init__(
            basic_conv2d=QuantizableBasicConv2d,
            inception=QuantizableInception,
            *args,
            **kwargs
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self._transform_input(x)
        x = self.quant(x)
        x, aux1, aux2 = self._forward(x)
        x = self.dequant(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted QuantizableGoogleNet always returns QuantizableGoogleNetOutputs Tuple")
            return QuantizableGoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in googlenet model

        Fuse conv+bn+relu/ conv+relu/conv+bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        for m in self.modules():
            if type(m) == QuantizableBasicConv2d:
                torch.quantization.fuse_modules(m, ["conv", "bn", "relu"], inplace=True)
