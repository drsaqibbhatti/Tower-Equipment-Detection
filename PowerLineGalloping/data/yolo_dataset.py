from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class YoloTxtDetectionDataset(Dataset):
    """
    Expects:
      root/
        images/train/*.jpg
        labels/train/*.txt   # YOLO: cls cx cy w h (normalized)
    """
    def __init__(self, root: str, split: str = "train", imgsz: int = 640):
        self.root = Path(root)
        self.split = split
        self.imgsz = imgsz

        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split

        self.images = [p for p in self.img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        self.images.sort()
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def _read_labels(self, img_path: Path, w: int, h: int):
        label_path = self.lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        rows = []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            # convert normalized YOLO -> pixel xyxy
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            rows.append([cls, x1, y1, x2, y2])
        return np.array(rows, dtype=np.float32)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(str(img_path))
        h, w = img.shape[:2]

        labels = self._read_labels(img_path, w=w, h=h)  # [N,5] cls,x1,y1,x2,y2

        # Minimal preprocessing: resize to square (you can swap letterbox if desired)
        img_resized = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img_rgb = img_resized[:, :, ::-1].astype(np.float32) / 255.0
        x = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).float()  # CHW

        y = torch.from_numpy(labels).float()
        return x, y