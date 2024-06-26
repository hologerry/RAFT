import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from core.pwc_correlation import ModuleCorrelation


class Decoder(nn.Module):
    def __init__(self, level):
        super().__init__()

        previous_channel = [None, None, 81 + 16 + 2 + 2, 81 + 24 + 2 + 2, 81 + 48 + 2 + 2, 81, None][level+1]
        current_channel = [None, None, 81 + 16 + 2 + 2, 81 + 24 + 2 + 2, 81 + 48 + 2 + 2, 81, None][level]

        self.corr = ModuleCorrelation()

        self.net_1 = nn.Sequential(
            nn.Conv2d(in_channels=current_channel, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.net_2 = nn.Sequential(
            nn.Conv2d(in_channels=current_channel + 48, out_channels=48, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.net_3 = nn.Sequential(
            nn.Conv2d(in_channels=current_channel + 48 + 48, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.net_4 = nn.Sequential(
            nn.Conv2d(in_channels=current_channel + 48 + 48 + 24, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.net_5 = nn.Sequential(
            nn.Conv2d(in_channels=current_channel + 48 + 48 + 24 + 16, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        if level < 5:
            self.up_flow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
            self.up_feat = nn.ConvTranspose2d(in_channels=previous_channel + 48 + 48 + 24 + 16, out_channels=2, kernel_size=4, stride=2, padding=1)
            self.backwarp_coefficient = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][level + 1]
            # https://github.com/NVlabs/PWC-Net/issues/119

        self.backwarp_grid = {}
        self.backwarp_partial = {}


    def backwarp(self, feat, flow):
        assert feat.size()[-2:] == flow.size()[-2:], f"feat.size() {feat.size()} flow.size() {flow.size()}"

        assert feat.device == flow.device
        if str(flow.shape) + '_' + str(flow.device) not in self.backwarp_grid:
            horizontal = torch.linspace(-1.0 + (1.0 / flow.shape[3]), 1.0 - (1.0 / flow.shape[3]), flow.shape[3], device=flow.device).view(1, 1, 1, -1).expand(-1, -1, flow.shape[2], -1)
            vertical = torch.linspace(-1.0 + (1.0 / flow.shape[2]), 1.0 - (1.0 / flow.shape[2]), flow.shape[2], device=flow.device).view(1, 1, -1, 1).expand(-1, -1, -1, flow.shape[3])

            self.backwarp_grid[str(flow.shape) + '_' + str(flow.device)] = torch.cat([ horizontal, vertical ], 1)


        if str(flow.shape) + '_' + str(flow.device) not in self.backwarp_partial:
            self.backwarp_partial[str(flow.shape) + '_' + str(flow.device)] = flow.new_ones([ flow.shape[0], 1, flow.shape[2], flow.shape[3] ])


        flow = torch.cat([ flow[:, 0:1, :, :] / ((feat.shape[3] - 1.0) / 2.0), flow[:, 1:2, :, :] / ((feat.shape[2] - 1.0) / 2.0) ], 1)
        cache = self.backwarp_partial[str(flow.shape) + '_' + str(flow.device)]
        # print(f"feat size {feat.size()} cache size {cache.size()}")
        assert feat.size()[-2:] == cache.size()[-2:]
        feat = torch.cat([ feat, cache], 1)

        output = F.grid_sample(input=feat, grid=(self.backwarp_grid[str(flow.shape) + '_' + str(flow.device)] + flow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

        mask = output[:, -1:, :, :]; mask[mask > 0.999] = 1.0; mask[mask < 1.0] = 0.0

        return output[:, :-1, :, :] * mask


    def forward(self, feat_1, feat_2, prev_level_out):
        if prev_level_out is None:
            feat_1_2_corr = self.corr(feat_1, feat_2)
            corr_volume = F.leaky_relu(feat_1_2_corr, negative_slope=0.1, inplace=False)
            feat_out = torch.cat([corr_volume], 1)

        else:
            upped_flow = self.up_flow(prev_level_out['flow'])
            upped_feat = self.up_feat(prev_level_out['feat'])
            warpped_feat_2 = self.backwarp(feat_2, upped_flow*self.backwarp_coefficient)
            feat_1_wap_2_corr = self.corr(feat_1, warpped_feat_2)
            corr_volume = F.leaky_relu(feat_1_wap_2_corr, negative_slope=0.1, inplace=False)

            feat_out = torch.cat([corr_volume, feat_1, upped_flow, upped_feat], dim=1)

        feat_out = torch.cat([self.net_1(feat_out), feat_out], dim=1)
        feat_out = torch.cat([self.net_2(feat_out), feat_out], dim=1)
        feat_out = torch.cat([self.net_3(feat_out), feat_out], dim=1)
        feat_out = torch.cat([self.net_4(feat_out), feat_out], dim=1)

        flow = self.net_5(feat_out)

        return {
            'flow': flow,
            'feat': feat_out
        }
