from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from unet_arch import UNet


UNET_INPUT_SIZE = 512
UNET_THRESHOLD = 0.5


class InferenceServiceError(Exception):
    pass


class MissingDependencyError(InferenceServiceError):
    def __init__(self, dependency: str, message: str | None = None):
        self.dependency = dependency
        super().__init__(message or f"Missing dependency: {dependency}")


class ModelUnavailableError(InferenceServiceError):
    def __init__(self, model: str, message: str | None = None):
        self.model = model
        super().__init__(message or f"Model unavailable: {model}")


@dataclass(frozen=True)
class InferenceResult:
    detections: list[dict[str, Any]]


class InferenceService:
    def __init__(
        self,
        *,
        backend_dir: Path | None = None,
        device: torch.device | None = None,
        yolo_weights: str = r"models/yolo-weights.pt",
        unet_weights_filename: str = "unet.pth",
    ) -> None:
        self._backend_dir = (backend_dir or Path(__file__).resolve().parent).resolve()
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Using device: {self._device}")

        # Resolve weights relative to backend_dir so it works regardless of cwd (incl. Docker)
        yolo_weights_path = self._resolve_existing_file(yolo_weights, kind="YOLO weights")
        self._yolo = YOLO(str(yolo_weights_path))
        self._unet: UNet | None = None
        try:
            weights_path = self._resolve_existing_file(unet_weights_filename, kind="UNet weights")

            model = UNet(in_channels=3, out_channels=2)
            state = torch.load(str(weights_path), map_location=self._device)
            model.load_state_dict(state)
            model.to(self._device)
            model.eval()
            self._unet = model
            print(f"[INFO] UNet loaded successfully: {weights_path}")
        except Exception as e:
            print(f"[WARN] Failed to load UNet: {e}")
            self._unet = None

    def _resolve_existing_file(self, path: str | Path, *, kind: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            if p.exists():
                return p
            raise FileNotFoundError(f"{kind} not found at absolute path: {p}")

        # Prefer explicit paths relative to backend_dir, then fall back to backend_dir/models.
        candidate_1 = (self._backend_dir / p)
        if candidate_1.exists():
            return candidate_1

        candidate_2 = (self._backend_dir / "models" / p)
        if candidate_2.exists():
            return candidate_2

        raise FileNotFoundError(
            f"{kind} not found. Tried: {candidate_1} and {candidate_2}"
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def yolo(self) -> YOLO:
        return self._yolo

    @property
    def unet_available(self) -> bool:
        return self._unet is not None

    def infer(self, model: str, pil_img: Image.Image) -> InferenceResult:
        model = model.lower().strip()
        if model == "yolo":
            return InferenceResult(detections=self._infer_yolo(pil_img))
        if model == "unet":
            return InferenceResult(detections=self._infer_unet(pil_img))
        raise ValueError(f"Unknown model: {model}")

    def _infer_yolo(self, pil_img: Image.Image) -> list[dict[str, Any]]:
        results = self._yolo(pil_img)
        r = results[0]

        detections: list[dict[str, Any]] = []
        for i in range(len(r.boxes)):
            det: dict[str, Any] = {
                "class": r.names[int(r.boxes[i].cls)],
                "conf": float(r.boxes[i].conf),
                "box": r.boxes[i].xyxy[0].tolist(),
            }
            if r.masks is not None:
                det["segments"] = r.masks.xyn[i].tolist()  # normalized
            detections.append(det)

        return detections

    def _pil_to_unet_input(self, pil_img: Image.Image) -> torch.Tensor:
        img_resized = pil_img.resize((UNET_INPUT_SIZE, UNET_INPUT_SIZE), Image.BILINEAR)
        img_np = np.asarray(img_resized, dtype=np.float32) / 255.0
        return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self._device)

    def _infer_unet(self, pil_img: Image.Image) -> list[dict[str, Any]]:
        if self._unet is None:
            raise ModelUnavailableError("unet", "UNet model not available on server")

        try:
            import cv2
        except ImportError as e:
            raise MissingDependencyError(
                "cv2",
                "OpenCV (cv2) is required for UNet contour extraction",
            ) from e

        orig_w, orig_h = pil_img.size
        inp = self._pil_to_unet_input(pil_img)

        with torch.no_grad():
            out = self._unet(inp)  # [1,2,H,W] logits
            probs = torch.sigmoid(out)[0].cpu().numpy()  # (2,H,W)

        class_names = ["fluid", "tumor"]
        detections: list[dict[str, Any]] = []

        for ch_idx, cname in enumerate(class_names):
            prob_map = probs[ch_idx]
            mask_bin = (prob_map > UNET_THRESHOLD).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cnt is None or cnt.shape[0] < 3:
                    continue
                cnt = cnt.squeeze()
                if cnt.ndim != 2 or cnt.shape[0] < 3:
                    continue

                xs = cnt[:, 0].astype(float) * (orig_w / UNET_INPUT_SIZE)
                ys = cnt[:, 1].astype(float) * (orig_h / UNET_INPUT_SIZE)

                x1, y1 = float(xs.min()), float(ys.min())
                x2, y2 = float(xs.max()), float(ys.max())

                seg = [[float(x) / orig_w, float(y) / orig_h] for x, y in zip(xs, ys)]

                mask_bool = mask_bin > 0
                conf = float(prob_map[mask_bool].mean()) if mask_bool.any() else 0.0

                detections.append(
                    {
                        "class": cname,
                        "conf": conf,
                        "box": [x1, y1, x2, y2],
                        "segments": seg,
                    }
                )

        return detections
