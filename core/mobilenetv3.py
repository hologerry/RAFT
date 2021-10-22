import math
import os
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models.mobilenetv2 import ConvBNActivation, _make_divisible


class SqueezeExcitation(nn.Module):
    # Implemented as described at Figure 4 of the MobileNetV3 paper
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(self, arch: str, \
        last_stage : int = 5, freeze_norm_layer: bool = False, \
        weights: str = None, norm_type: str = 'batch', **kwargs: Any) -> None:
        """
        MobileNet V3 main class
        """
        super().__init__()
        self.last_stage = last_stage
        assert 2 <= self.last_stage <= 5, f"'last_stage' for MobileNetV3 must be in [2, 5]."

        inverted_residual_setting, _ =  MobileNetV3.mobilenet_v3_conf(arch, **kwargs)
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        # record the out_channels for different stage
        stride_mapping, stride_cursor = dict(), 2
        for i, b in enumerate(inverted_residual_setting):
            stride_cursor *= b.stride
            stride_mapping[stride_cursor] = (i+1, b.out_channels)

        if norm_type == 'batch':
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        elif norm_type == 'instance':
            norm_layer = partial(nn.InstanceNorm2d, eps=0.001, momentum=0.01)
        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        s_ = 2
        for i, cnf in enumerate(inverted_residual_setting):
            s_ *= cnf.stride
            if s_ <= 2**self.last_stage:
                layers.append(InvertedResidual(cnf, norm_layer))
            else:
                break
        else:
            # building last several layers
            lastconv_input_channels = inverted_residual_setting[-1].out_channels
            lastconv_output_channels = 6 * lastconv_input_channels
            layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                           norm_layer=norm_layer, activation_layer=nn.Hardswish))

            # do not forget the last several layers
            stride_mapping[stride_cursor] = (len(inverted_residual_setting) + 1, lastconv_output_channels)

        self.features = nn.Sequential(*layers)
        self.out_channels, self.checkpoints = dict(), dict()
        for s in stride_mapping:
            checkpoint, out_channels = stride_mapping[s]
            self.out_channels[f'C{int(math.log2(s))}'] = out_channels
            self.checkpoints[checkpoint] = s

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.zeros_(m.bias)

        if weights is not None:
            self.freeze_norm_layer = freeze_norm_layer
            self.load_state_dict(torch.load(weights), strict=False)
        else:
            self.freeze_norm_layer = False

            # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def train(self, mode : bool = True):
        super().train(mode)
        if self.freeze_norm_layer:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        feats = dict()

        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.checkpoints:
                stride = self.checkpoints[i]

                k = f'C{int(math.log2(stride))}'
                feats[k] = x

        return feats
    
    @property
    def size_divisible_by_n(self):
        return 0

    @staticmethod
    def mobilenet_v3_conf(arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False,
                        **kwargs: Any):
        reduce_divider = 2 if reduced_tail else 1
        dilation = 2 if dilated else 1

        bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
        adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

        if arch == "mobilenet_v3_large":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
                bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
                bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
                bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
                bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
                bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
                bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
                bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
                bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
                bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            ]
            last_channel = adjust_channels(1280 // reduce_divider)  # C5
        elif arch == "mobilenet_v3_small":
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
                bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
                bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
                bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
                bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
                bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
                bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
                bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            ]
            last_channel = adjust_channels(1024 // reduce_divider)  # C5
        else:
            raise ValueError("Unsupported model type {}".format(arch))

        return inverted_residual_setting, last_channel


if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    net = MobileNetV3('mobilenet_v3_small', last_stage=5, norm_type='instance').cuda()
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (3, 160, 96), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        data = torch.randn((2, 3, 224, 224)).cuda()
        outs = net(data)

        print("type outs", type(outs))
        print("")

        for k in outs:
            print(k, outs[k].shape)
