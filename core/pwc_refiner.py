import torch
import torch.nn as nn


class Refiner(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels=237, out_channels=48, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=24, kernel_size=3, stride=1, padding=16, dilation=16),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        )

    def forward(self, input):
        output = self.main(input)
        return output
