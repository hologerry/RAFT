import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.corr import AlternateCorrBlock, CorrBlock
from core.extractor import BasicEncoder, SmallEncoder
from core.mobilenetv3 import MobileNetV3
from core.pwc_decoder import Decoder
from core.pwc_refiner import Refiner
from core.update import BasicUpdateBlock, SmallUpdateBlock, TinyUpdateBlock
from core.utils.utils import bilinear_sampler, coords_grid, upflow8, upflow4

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class PWCTiny(nn.Module):
    def __init__(self, args):
        super(PWCTiny, self).__init__()
        self.args = args

        self.extractor = MobileNetV3('mobilenet_v3_small', last_stage=5, norm_type='instance')
        self.decoder_2 = Decoder(level=2)
        self.decoder_3 = Decoder(level=3)
        self.decoder_4 = Decoder(level=4)
        self.decoder_5 = Decoder(level=5)
        self.refiner = Refiner()

    def is_training(self):
        return self.args.mode == 'train'

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        """ Estimate optical flow between pair of frames """
        image1, image2 = torch.chunk(x, 2, dim=1)
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        feats1 = self.extractor(image1)
        feats2 = self.extractor(image2)

        estimate_out = self.decoder_5(feats1['C5'], feats2['C5'], None)
        flow_5 = estimate_out['flow']
        estimate_out = self.decoder_4(feats1['C4'], feats2['C4'], estimate_out)
        flow_4 = estimate_out['flow']
        estimate_out = self.decoder_3(feats1['C3'], feats2['C3'], estimate_out)
        flow_3 = estimate_out['flow']
        estimate_out = self.decoder_2(feats1['C2'], feats2['C2'], estimate_out)

        flow_2 = estimate_out['flow'] + self.refiner(estimate_out['feat'])

        flow_out = upflow4(flow_2)

        if self.is_training():
            return flow_out, flow_2, flow_3, flow_4, flow_5
        else:
            
            return flow_out * 20.0


if __name__ == '__main__':
    import argparse

    from ptflops import get_model_complexity_info

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='pwc_tiny', help="name your experiment")
    parser.add_argument('--mode', default='train', help="")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    net = PWCTiny(args).cuda()
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (6, 160, 96), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        data = torch.randn((2, 6, 224, 224)).cuda()
        out = net(data)


