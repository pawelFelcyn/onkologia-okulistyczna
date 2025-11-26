import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import pandas as pd
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

class UNetDataset(Dataset):
    def __init__(self, csv_path, root_dir="", transforms=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transforms = transforms
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path  = os.path.join(self.root_dir, row["image_path"])
        mask_path = os.path.join(self.root_dir, row["mask_path"])

        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path).convert("L")

        if self.transforms:
            img = self.transforms(img)

        img_tensor = self.to_tensor(img)
        
        mask_np = np.array(mask, dtype=np.int64)
        tolerance = 10
        chan0 = np.zeros_like(mask_np, dtype=np.float32)
        chan0[(mask_np >= 127 - tolerance) & (mask_np <= 127 + tolerance)] = 1.0
        chan1 = np.zeros_like(mask_np, dtype=np.float32)
        chan1[mask_np >= 255 - tolerance] = 1.0
        mask_tensor = np.stack([chan0, chan1], axis=0)
        mask_tensor = torch.from_numpy(mask_tensor)

        return img_tensor, mask_tensor

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, base=64):
        super().__init__()

        # --- DOWN ---
        self.conv1 = self.double_conv(in_channels, base)
        self.conv2 = self.double_conv(base, base*2)
        self.conv3 = self.double_conv(base*2, base*4)
        self.conv4 = self.double_conv(base*4, base*8)
        self.conv5 = self.double_conv(base*8, base*16)

        # --- POOL ---
        self.pool = nn.MaxPool2d(2)

        # --- UP ---
        self.up1 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(base*8,  base*4, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(base*4,  base*2, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(base*2,  base,   2, stride=2)

        # --- UP CONV ---
        self.uconv1 = self.double_conv(base*16, base*8)
        self.uconv2 = self.double_conv(base*8,  base*4)
        self.uconv3 = self.double_conv(base*4,  base*2)
        self.uconv4 = self.double_conv(base*2,  base)

        # --- OUTPUT ---
        self.out = nn.Conv2d(base, out_channels, kernel_size=1)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
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

    def pad_and_concat(self, up, skip):
        diffY = skip.size()[2] - up.size()[2]
        diffX = skip.size()[3] - up.size()[3]

        up = F.pad(up, [diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2])

        return torch.cat([skip, up], dim=1)
    
    def __get_run_dir(self):
        base_dir = "runs_unet"
        prefix = "run"
        os.makedirs(base_dir, exist_ok=True)
        i = 1
        while True:
            candidate = os.path.join(base_dir, f"{prefix}{i}")
            if not os.path.exists(candidate):
                return candidate
            i += 1
    
    def train_model(self,
          train_loader,
          val_loader,
          num_epochs=20,
          lr=1e-4,
          device=None):

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        run_dir = self.__get_run_dir()
        os.makedirs(run_dir, exist_ok=True)
        weights_dir = os.path.join(run_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        self.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            epoch_dir = os.path.join(run_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            epoch_data = {
                'epoch_number': epoch
            }
            self.train()
            train_loss = 0.0
            for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} - Training", leave=False):
                imgs = imgs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                preds = self(imgs)
                loss = criterion(preds, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            train_loss /= len(train_loader)
            epoch_data['train_loss'] = train_loss


            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} - Validation", leave=False):
                    imgs = imgs.to(device)
                    masks = masks.to(device)

                    preds = self(imgs)
                    loss = criterion(preds, masks)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            epoch_data['val_loss'] = val_loss
            
            with open(os.path.join(epoch_dir, "epoch_data.json"), "w") as f:
                json.dump(epoch_data, f)
            torch.save(self.state_dict(), os.path.join(weights_dir, 'last.pth'))
            print("✓ saved last.pth")

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.state_dict(), os.path.join(weights_dir, 'best.pth'))
                print("✓ saved best_unet.pth")

        print("\nFinished training. Best val loss:", best_val_loss)
        
    def save(self, path):
        torch.save(self.state_dict(), path)