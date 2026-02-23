from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model.model import ObjNano
from utils.preproc import letterbox, bgr_to_tensor
from utils.nms import multiclass_nms
from utils.postproc import scale_boxes_back, draw_detections
from utils.io import iter_images

def load_model(weights: str, device: str, half: bool, num_classes: int = 80):
    model = ObjNano(num_classes=num_classes)
    ckpt = torch.load(weights, map_location="cpu")

    # support common checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    # strip 'module.' if needed
    clean = {}
    for k, v in state.items():
        clean[k.replace("module.", "")] = v

    model.load_state_dict(clean, strict=False)
    model.to(device).eval()
    if half and device.startswith("cuda"):
        model.half()
    return model

@torch.no_grad()
def decode_model_output(pred, conf_thres: float):
    """
    Decode model output to boxes, scores, and class IDs.
    
    Model output format: [B, 4+nc, N] where:
      - First 4 channels: xyxy box coordinates (already in pixel space)
      - Remaining nc channels: sigmoid-activated class scores
    
    Returns: boxes_xyxy (Tensor [N,4]), scores (Tensor [N]), classes (Tensor [N])
    """
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    
    # pred: [B, 4+nc, N]
    if pred.ndim == 3:
        pred = pred[0]  # Take first batch: [4+nc, N]
        pred = pred.transpose(0, 1)  # [N, 4+nc]
    
    # Now pred is [N, 4+nc]
    boxes = pred[:, :4]  # [N, 4] - xyxy format, already in pixel coords
    cls_scores = pred[:, 4:]  # [N, nc] - sigmoid-activated class scores
    
    # Get best class and score for each detection
    scores, cls_ids = cls_scores.max(dim=1)  # [N], [N]
    
    # Filter by confidence threshold
    keep = scores >= conf_thres
    boxes = boxes[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]
    
    return boxes, scores, cls_ids

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="image file or folder")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--half", action="store_true")
    p.add_argument("--out", default="outputs")
    p.add_argument("--names", default="", help="comma-separated class names (optional)")
    p.add_argument("--num_classes", type=int, default=80, help="number of classes")
    return p.parse_args()

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    names = None
    if args.names.strip():
        items = [s.strip() for s in args.names.split(",")]
        names = {i: n for i, n in enumerate(items)}

    model = load_model(args.weights, device=device, half=args.half, num_classes=args.num_classes)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in iter_images(args.source):
        img0 = cv2.imread(path)
        if img0 is None:
            print(f"Skip unreadable: {path}")
            continue

        h0, w0 = img0.shape[:2]
        img, r, pad = letterbox(img0, new_shape=args.imgsz)
        x = bgr_to_tensor(img)  # CHW float32
        x = torch.from_numpy(x).unsqueeze(0)  # [1,3,H,W]
        x = x.to(device)
        if args.half and device.startswith("cuda"):
            x = x.half()

        pred = model(x)  # your forward
        boxes, scores, cls_ids = decode_model_output(pred, conf_thres=args.conf)

        # NMS in letterbox coords
        boxes, scores, cls_ids = multiclass_nms(
            boxes_xyxy=boxes,
            scores=scores,
            classes=cls_ids,
            iou_thres=args.iou,
            conf_thres=args.conf,
            max_det=300,
        )

        # scale boxes back to original image
        boxes = scale_boxes_back(boxes, ratio=r, pad=pad, orig_shape=(h0, w0))

        # draw + save
        img_vis = draw_detections(
            img0,
            boxes.cpu().numpy(),
            scores.cpu().numpy(),
            cls_ids.cpu().numpy(),
            names=names,
        )
        save_path = out_dir / (Path(path).stem + "_pred.jpg")
        cv2.imwrite(str(save_path), img_vis)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()