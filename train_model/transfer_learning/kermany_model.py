"""
kermany_model.py
================
Model architecture for Kermany OCT pretraining:
  - double_conv()       – convolutional block (identical to unet_utils.py)
  - UNetEncoder         – U-Net encoder (conv1–conv5)
  - KermanyClassifier   – encoder + classification head (4 classes)

Imported by train_kermany.py and eval_kermany.py.
In Stage B the encoder weights can be loaded directly into U-Net:
    model.encoder.load_state_dict(torch.load("encoder_kermany_pretrained.pth"))
"""

import torch
import torch.nn as nn

from kermany_dataset import NUM_CLASSES

# ---------------------------------------------------------------------------
# Convolutional block
# ---------------------------------------------------------------------------

def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Two Conv2d-BN-ReLU blocks. Identical to UNet.double_conv() in unet_utils.py,
    ensuring weight compatibility when transferring to the U-Net.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# U-Net Encoder
# ---------------------------------------------------------------------------

class UNetEncoder(nn.Module):
    """
    U-Net encoder: 5 double_conv blocks with MaxPool2d downsampling.

    Architecture is identical to UNet(base=64) in unet_utils.py.
    Can be plugged directly into U-Net as model.encoder.

    Output shape: (B, base*16, H/16, W/16)  →  (B, 1024, H/16, W/16) for base=64
    """
    def __init__(self, in_channels: int = 3, base: int = 64):
        super().__init__()
        self.conv1 = double_conv(in_channels, base)       # 64
        self.conv2 = double_conv(base,        base * 2)   # 128
        self.conv3 = double_conv(base * 2,    base * 4)   # 256
        self.conv4 = double_conv(base * 4,    base * 8)   # 512
        self.conv5 = double_conv(base * 8,    base * 16)  # 1024
        self.pool  = nn.MaxPool2d(2)
        self.out_channels = base * 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(self.pool(x))
        x = self.conv3(self.pool(x))
        x = self.conv4(self.pool(x))
        x = self.conv5(self.pool(x))
        return x


# ---------------------------------------------------------------------------
# Classifier: Encoder + head
# ---------------------------------------------------------------------------

class KermanyClassifier(nn.Module):
    """
    U-Net encoder + classification head for Kermany OCT pretraining.

    Architecture:
        input (B, 3, 224, 224)
          ↓
        UNetEncoder                     →  (B, 1024, 14, 14)
          ↓
        AdaptiveAvgPool2d(1)            →  (B, 1024, 1, 1)
          ↓
        Flatten                         →  (B, 1024)
          ↓
        Linear(1024 → num_classes)      →  (B, 4)  logits
    """
    def __init__(self, in_channels: int = 3, base: int = 64,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = UNetEncoder(in_channels=in_channels, base=base)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.head    = nn.Linear(self.encoder.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)           # (B, 1024, H', W')
        pooled   = self.pool(features)       # (B, 1024, 1, 1)
        flat     = torch.flatten(pooled, 1)  # (B, 1024)
        return self.head(flat)               # (B, num_classes) logits
