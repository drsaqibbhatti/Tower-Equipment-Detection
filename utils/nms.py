from __future__ import annotations
import torch
from torchvision.ops import nms

def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    # xywh: [...,4] with x,y,w,h (center-based)
    x, y, w, h = xywh.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)

@torch.no_grad()
def multiclass_nms(
    boxes_xyxy: torch.Tensor,   # [N,4]
    scores: torch.Tensor,       # [N]
    classes: torch.Tensor,      # [N] int
    iou_thres: float = 0.5,
    conf_thres: float = 0.25,
    max_det: int = 300,
):
    keep = scores >= conf_thres
    boxes_xyxy = boxes_xyxy[keep]
    scores = scores[keep]
    classes = classes[keep]

    if boxes_xyxy.numel() == 0:
        return boxes_xyxy, scores, classes

    kept_idx_all = []
    for c in classes.unique():
        idx = (classes == c).nonzero(as_tuple=False).squeeze(1)
        kept = nms(boxes_xyxy[idx], scores[idx], iou_thres)
        kept_idx_all.append(idx[kept])

    kept_idx = torch.cat(kept_idx_all, dim=0)
    kept_idx = kept_idx[scores[kept_idx].argsort(descending=True)]
    kept_idx = kept_idx[:max_det]

    return boxes_xyxy[kept_idx], scores[kept_idx], classes[kept_idx]