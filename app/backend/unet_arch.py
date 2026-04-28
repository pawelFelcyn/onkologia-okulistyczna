import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 2, base: int = 64):
        super().__init__()

        # --- DOWN ---
        self.conv1 = self.double_conv(in_channels, base)
        self.conv2 = self.double_conv(base, base * 2)
        self.conv3 = self.double_conv(base * 2, base * 4)
        self.conv4 = self.double_conv(base * 4, base * 8)
        self.conv5 = self.double_conv(base * 8, base * 16)

        # --- POOL ---
        self.pool = nn.MaxPool2d(2)

        # --- UP ---
        self.up1 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)

        # --- UP CONV ---
        self.uconv1 = self.double_conv(base * 16, base * 8)
        self.uconv2 = self.double_conv(base * 8, base * 4)
        self.uconv3 = self.double_conv(base * 4, base * 2)
        self.uconv4 = self.double_conv(base * 2, base)

        # --- OUTPUT ---
        self.out = nn.Conv2d(base, out_channels, kernel_size=1)

    def double_conv(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DOWN
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        # UP 1
        u1 = self.up1(x5)
        u1 = self.pad_and_concat(u1, x4)
        u1 = self.uconv1(u1)

        # UP 2
        u2 = self.up2(u1)
        u2 = self.pad_and_concat(u2, x3)
        u2 = self.uconv2(u2)

        # UP 3
        u3 = self.up3(u2)
        u3 = self.pad_and_concat(u3, x2)
        u3 = self.uconv3(u3)

        # UP 4
        u4 = self.up4(u3)
        u4 = self.pad_and_concat(u4, x1)
        u4 = self.uconv4(u4)

        return self.out(u4)

    def pad_and_concat(self, up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diff_y = skip.size()[2] - up.size()[2]
        diff_x = skip.size()[3] - up.size()[3]

        up = F.pad(
            up,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        return torch.cat([skip, up], dim=1)
