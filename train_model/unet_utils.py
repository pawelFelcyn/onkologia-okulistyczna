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
from torchmetrics import ConfusionMatrix

def metrics_from_confusion_matrix(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]

    # Avoid division by zero
    eps = 1e-7

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    dice = f1
    iou = tp / (tp + fp + fn + eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "dice": dice,
        "iou": iou
    }

def get_confucion_matrcies(masks: torch.Tensor, preds: torch.Tensor):
    #first channel is for fluid, second for tumor
    preds_fluid = preds[:, 0, :, :]
    preds_tumor = preds[:, 1, :, :]
    masks_fluid = masks[:, 0, :, :]
    masks_tumor = masks[:, 1, :, :]
    
    cm = ConfusionMatrix(task="binary", num_classes=2)
    fluid_cm = cm(preds_fluid, masks_fluid)
    tumor_cm = cm(preds_tumor, masks_tumor)
    return np.array(fluid_cm), np.array(tumor_cm)

def get_metrics(masks: torch.Tensor, preds: torch.Tensor):
    fluid_cm, tumor_cm = get_confucion_matrcies(masks, preds)
    fluid_metrics = metrics_from_confusion_matrix(fluid_cm)
    tumor_metrics = metrics_from_confusion_matrix(tumor_cm)
    return fluid_metrics, fluid_cm, tumor_metrics, tumor_cm
    

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
        tumor_mask_path = os.path.join(self.root_dir, row["tumor_mask_path"])
        fluid_mask_path = os.path.join(self.root_dir, row["fluid_mask_path"])

        img = Image.open(img_path).convert("RGB")

        tumor_mask = Image.open(tumor_mask_path).convert("L")
        fluid_mask = Image.open(fluid_mask_path).convert("L")

        if self.transforms:
            img = self.transforms(img)

        img_tensor = self.to_tensor(img)
        
        tumor_mask_np = np.array(tumor_mask, dtype=np.int64)
        fluid_mask_np = np.array(fluid_mask, dtype=np.int64)
       
        mask_tensor = torch.from_numpy(np.stack([fluid_mask_np, tumor_mask_np], axis=0))

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
    
    def __get_test_run_dir(self):
        base_dir = "runs_unet"
        prefix = "test_run"
        os.makedirs(base_dir, exist_ok=True)
        i = 1
        while True:
            candidate = os.path.join(base_dir, f"{prefix}{i}")
            if not os.path.exists(candidate):
                return candidate
            i += 1
            
    def __cn_to_dict(self, cn):
        return {
            "TP": int(cn[1][1]),
            "TN": int(cn[0][0]),
            "FP": int(cn[0][1]),
            "FN": int(cn[1][0])
        }
    
    def test_model(self, test_loader, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_results_dir = self.__get_test_run_dir()
        os.makedirs(test_results_dir, exist_ok=True)
        
        self.to(device)

        self.eval()
        fluid_cm = None
        tumor_cm = None
        with torch.no_grad():
            for imgs, masks in tqdm(test_loader, desc=f"Testing: ", leave=False):
                imgs = imgs.to(device)
                masks = masks.to(device)
                preds = self(imgs)
                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()
                fluid_cm_local, tumor_cm_local = get_confucion_matrcies(masks, preds)
                if fluid_cm is None:
                    fluid_cm = fluid_cm_local
                else:
                    fluid_cm += fluid_cm_local
                if tumor_cm is None:
                    tumor_cm = tumor_cm_local
                else:
                    tumor_cm += tumor_cm_local
                    
        fluid_metrics = metrics_from_confusion_matrix(fluid_cm)
        tumor_metrics = metrics_from_confusion_matrix(tumor_cm)
        
        with open(os.path.join(test_results_dir, "fluid_metrics.json"), "w") as f:
            json.dump(fluid_metrics, f)
        with open(os.path.join(test_results_dir, "tumor_metrics.json"), "w") as f:
            json.dump(tumor_metrics, f)
            
        with open(os.path.join(test_results_dir, "fluid_cm.json"), "w") as f:
            json.dump(self.__cn_to_dict(fluid_cm), f)
        with open(os.path.join(test_results_dir, "tumor_cm.json"), "w") as f:
            json.dump(self.__cn_to_dict(tumor_cm), f)
        
        return fluid_metrics, fluid_cm, tumor_metrics, tumor_cm
            
    def save(self, path):
        torch.save(self.state_dict(), path)