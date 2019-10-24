import torch
import unittest
import torchvision

from common_utils import TestCase, map_nested_tensor_object
from collections import OrderedDict
from itertools import product
import torch
import numpy as np
from torchvision import models
import unittest
import traceback
import random


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


quantizable_models = ['googlenet', 'inception_v3', 'shufflenet_v2_x0_5',
                           'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
                           'shufflenet_v2_x2_0']

scriptable_quantizable_models = ['googlenet', 'inception_v3']

class ModelTester(TestCase):
    def check_script(self, model, name):
        if name not in scriptable_quantizable_models:
            return
        scriptable = True
        msg = ""
        try:
            torch.jit.script(model)
        except Exception as e:
            tb = traceback.format_exc()
            scriptable = False
            msg = str(e) + str(tb)
        self.assertTrue(scriptable, msg)

    def _test_classification_model(self, name, input_shape):
        # passing num_class equal to a number other than 1000 helps in making the test
        # more enforcing in nature

        for eval in [True, False]:
            model = torchvision.models.quantization.__dict__[name](pretrained_float_model=False)
            if eval:
                model.eval()
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            else:
                model.train()
                model.qconfig = torch.quantization.default_qat_qconfig


            model.fuse_model()
            if eval:
                torch.quantization.prepare(model, inplace=True)
            else:
                torch.quantization.prepare_qat(model, inplace=True)
                model.eval()
            torch.quantization.convert(model, inplace=True)
            # Ensure that quantized model runs successfully
            x = torch.rand(input_shape)
            out = model(x)

        self.check_script(model, name)


for model_name in quantizable_models:
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        input_shape = (1, 3, 224, 224)
        if model_name in ['inception_v3']:
            input_shape = (1, 3, 299, 299)
        self._test_classification_model(model_name, input_shape)

    setattr(ModelTester, "test_" + model_name, do_test)


if __name__ == '__main__':
    unittest.main()
