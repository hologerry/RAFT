import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .corr import AlternateCorrBlock, CorrBlock
from .extractor import BasicEncoder, SmallEncoder
from .mobilenetv3 import MobileNetV3
from .update import BasicUpdateBlock, SmallUpdateBlock, TinyUpdateBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8

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


class RAFTTiny(nn.Module):
    def __init__(self, args):
        super(RAFTTiny, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 16
        self.context_dim = 8
        args.corr_levels = 4
        args.corr_radius = 3

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block

        self.fnet = MobileNetV3('mobilenet_v3_small', last_stage=3, norm_type='instance')
        self.cnet = MobileNetV3('mobilenet_v3_small', last_stage=3, norm_type='instance')
        self.update_block = TinyUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=8, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # image1, image2 = torch.chunk(x, 2, dim=1)
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        batch_dim = image1.size(0)
        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            image = torch.cat([image1, image2], dim=0)
            feats = self.fnet(image)
            fmap = feats['C3']
            fmap1, fmap2 = torch.split(fmap, [batch_dim, batch_dim], dim=0)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet_feats = self.cnet(image1)
            cnet = cnet_feats['C3']
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)  # 16
            inp = torch.relu(inp)  # 8

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume, (batch, ht, wd, 1, ht, wd)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions


if __name__ == '__main__':
    import argparse

    from ptflops import get_model_complexity_info

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    net = RAFTTiny(args)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (6, 160, 96), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        data = torch.randn((2, 6, 224, 224))
        out = net(data)


