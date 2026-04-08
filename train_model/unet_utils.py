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
from torch.utils.tensorboard import SummaryWriter


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


def get_confusion_matrices(masks: torch.Tensor, preds: torch.Tensor):
    device = masks.device
    preds_fluid = preds[:, 0, :, :]
    preds_tumor = preds[:, 1, :, :]
    masks_fluid = masks[:, 0, :, :]
    masks_tumor = masks[:, 1, :, :]

    cm = ConfusionMatrix(task="binary", num_classes=2).to(device)
    fluid_cm = cm(preds_fluid, masks_fluid)
    tumor_cm = cm(preds_tumor, masks_tumor)
    return np.array(fluid_cm.cpu()), np.array(tumor_cm.cpu())


def get_metrics(masks: torch.Tensor, preds: torch.Tensor):
    fluid_cm, tumor_cm = get_confusion_matrices(masks, preds)
    fluid_metrics = metrics_from_confusion_matrix(fluid_cm)
    tumor_metrics = metrics_from_confusion_matrix(tumor_cm)
    return fluid_metrics, fluid_cm, tumor_metrics, tumor_cm


class UNetDataset(Dataset):
    def __init__(self, csv_path, root_dir="", transforms=None, imgsz=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transforms = transforms
        self.imgsz = imgsz
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(self.root_dir, row["image_path"])
        tumor_mask_path = os.path.join(self.root_dir, row["tumor_mask_path"])
        fluid_mask_path = os.path.join(self.root_dir, row["fluid_mask_path"])

        img = Image.open(img_path).convert("RGB")

        tumor_mask = Image.open(tumor_mask_path).convert("L")
        fluid_mask = Image.open(fluid_mask_path).convert("L")

        if self.imgsz:
            # Resize image with bilinear interpolation; masks with nearest to preserve label values
            img = img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
            tumor_mask = tumor_mask.resize((self.imgsz, self.imgsz), Image.NEAREST)
            fluid_mask = fluid_mask.resize((self.imgsz, self.imgsz), Image.NEAREST)

        if self.transforms:
            img = self.transforms(img)

        img_tensor = self.to_tensor(img)

        tumor_mask_np = np.array(tumor_mask, dtype=np.float32) / 255.0
        fluid_mask_np = np.array(fluid_mask, dtype=np.float32) / 255.0

        mask_tensor = torch.from_numpy(
            np.stack([fluid_mask_np, tumor_mask_np], axis=0))

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

    def get_encoder_blocks(self):
        return [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

    def __get_run_dir(self, run_name=None):
        base_dir = "runs_unet"
        prefix = "run"
        os.makedirs(base_dir, exist_ok=True)
        if run_name:
            candidate = os.path.join(base_dir, run_name)
            if os.path.exists(candidate):
                raise FileExistsError(
                    f"Run directory already exists: {candidate}. Choose a different --run_name or resume explicitly."
                )
            return candidate
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
                    device=None,
                    freeze_encoder=False,
                    run_name=None,
                    run_meta=None):

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        run_dir = self.__get_run_dir(run_name=run_name)
        os.makedirs(run_dir, exist_ok=True)
        weights_dir = os.path.join(run_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        if run_meta is not None:
            with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
                json.dump(run_meta, f, indent=2)

        self.to(device)

        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

        criterion = nn.BCEWithLogitsLoss()
        trainable_parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_parameters, lr=lr)
        encoder_blocks = self.get_encoder_blocks()

        if freeze_encoder:
            trainable_count = sum(p.numel() for p in trainable_parameters)
            print(f"[INFO] Training decoder/head only. Trainable parameters: {trainable_count:,}")

        best_val_dice = -1.0
        best_tumor_dice = -1.0
        best_fluid_dice = -1.0

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            epoch_dir = os.path.join(run_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            epoch_data = {'epoch_number': epoch}

            # ── TRAIN ──────────────────────────────────────────────────────
            self.train()
            if freeze_encoder:
                for block in encoder_blocks:
                    block.eval()

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

            # ── VAL ────────────────────────────────────────────────────────
            self.eval()
            val_loss = 0.0
            val_fluid_cm = None
            val_tumor_cm = None
            with torch.no_grad():
                for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} - Validation", leave=False):
                    imgs = imgs.to(device)
                    masks = masks.to(device)

                    preds = self(imgs)
                    loss = criterion(preds, masks)
                    val_loss += loss.item()

                    preds_bin = (torch.sigmoid(preds) > 0.5).float()
                    f_cm, t_cm = get_confusion_matrices(masks, preds_bin)
                    val_fluid_cm = f_cm if val_fluid_cm is None else val_fluid_cm + f_cm
                    val_tumor_cm = t_cm if val_tumor_cm is None else val_tumor_cm + t_cm

            val_loss /= len(val_loader)
            epoch_data['val_loss'] = val_loss

            fluid_m = metrics_from_confusion_matrix(val_fluid_cm)
            tumor_m = metrics_from_confusion_matrix(val_tumor_cm)
            val_dice_macro = (fluid_m['dice'] + tumor_m['dice']) / 2.0
            val_iou_macro = (fluid_m['iou'] + tumor_m['iou']) / 2.0

            epoch_data.update({
                'val_fluid_dice': fluid_m['dice'], 'val_fluid_iou': fluid_m['iou'],
                'val_tumor_dice': tumor_m['dice'], 'val_tumor_iou': tumor_m['iou'],
                'val_dice_macro': val_dice_macro,  'val_iou_macro':  val_iou_macro,
            })

            with open(os.path.join(epoch_dir, "epoch_data.json"), "w") as f:
                json.dump(epoch_data, f, indent=2)

            # ── TENSORBOARD ────────────────────────────────────────────────
            writer.add_scalar("Loss/train",         train_loss,     epoch)
            writer.add_scalar("Loss/val",           val_loss,       epoch)
            writer.add_scalar("Dice/val_fluid",     fluid_m['dice'], epoch)
            writer.add_scalar("Dice/val_tumor",     tumor_m['dice'], epoch)
            writer.add_scalar("Dice/val_macro",     val_dice_macro, epoch)
            writer.add_scalar("IoU/val_fluid",      fluid_m['iou'],  epoch)
            writer.add_scalar("IoU/val_tumor",      tumor_m['iou'],  epoch)
            writer.add_scalar("IoU/val_macro",      val_iou_macro,  epoch)
            writer.add_scalar("Recall/val_fluid",   fluid_m['recall'],  epoch)
            writer.add_scalar("Recall/val_tumor",   tumor_m['recall'],  epoch)
            writer.add_scalar("Precision/val_fluid",
                              fluid_m['precision'], epoch)
            writer.add_scalar("Precision/val_tumor",
                              tumor_m['precision'], epoch)

            torch.save(self.state_dict(), os.path.join(
                weights_dir, 'last.pth'))
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Dice fluid: {fluid_m['dice']:.4f} tumor: {tumor_m['dice']:.4f} macro: {val_dice_macro:.4f}")

            if val_dice_macro > best_val_dice:
                best_val_dice = val_dice_macro
                torch.save(self.state_dict(), os.path.join(
                    weights_dir, 'best.pth'))
                print(
                    f"✓ saved best.pth  (val_dice_macro={best_val_dice:.4f})")

            if tumor_m['dice'] > best_tumor_dice:
                best_tumor_dice = tumor_m['dice']
                torch.save(self.state_dict(), os.path.join(
                    weights_dir, 'best_tumor.pth'))
                print(
                    f"✓ saved best_tumor.pth  (val_tumor_dice={best_tumor_dice:.4f})")

            if fluid_m['dice'] > best_fluid_dice:
                best_fluid_dice = fluid_m['dice']
                torch.save(self.state_dict(), os.path.join(
                    weights_dir, 'best_fluid.pth'))
                print(
                    f"✓ saved best_fluid.pth  (val_fluid_dice={best_fluid_dice:.4f})")

        writer.close()
        print(
            f"\nTraining complete. Best val Dice (macro): {best_val_dice:.4f} | "
            f"Best tumor Dice: {best_tumor_dice:.4f} | Best fluid Dice: {best_fluid_dice:.4f}"
        )
        return {
            "run_dir": run_dir,
            "weights_dir": weights_dir,
            "best_val_dice": float(best_val_dice),
            "best_tumor_dice": float(best_tumor_dice),
            "best_fluid_dice": float(best_fluid_dice),
        }

    def __get_test_run_dir(self, run_name: str | None = None):
        base_dir = "runs_unet"
        os.makedirs(base_dir, exist_ok=True)

        if run_name:
            safe_name = os.path.basename(run_name)
            if safe_name != run_name:
                raise ValueError(f"Invalid run_name (must be a folder name, not a path): {run_name}")
            candidate = os.path.join(base_dir, safe_name)
            if os.path.exists(candidate):
                raise FileExistsError(
                    f"Test run directory already exists: {candidate}. Choose a different run_name."
                )
            return candidate

        prefix = "test_run"
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

    def test_model(self, test_loader, device=None, run_name: str | None = None):
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        test_results_dir = self.__get_test_run_dir(run_name=run_name)
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
                fluid_cm_local, tumor_cm_local = get_confusion_matrices(
                    masks, preds)
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
