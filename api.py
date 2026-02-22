from __future__ import annotations
import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from model.model import ObjNano
from utils.preproc import letterbox, bgr_to_tensor
from utils.nms import multiclass_nms
from utils.postproc import scale_boxes_back
from predict import decode_model_output  # reuse

app = FastAPI(title="Tower Defect Detection API (Pure PyTorch)")

WEIGHTS = "weights/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ = 960

# example class names
NAMES = {0: "crack", 1: "corrosion", 2: "loose_component", 3: "surface_damage"}

def load_model():
    model = ObjNano(num_classes=len(NAMES))
    ckpt = torch.load(WEIGHTS, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    model.to(DEVICE).eval()
    return model

model = load_model()

class Det(BaseModel):
    cls: int
    label: str
    conf: float
    xyxy: list[float]

class Resp(BaseModel):
    detections: list[Det]

def bytes_to_bgr(b: bytes):
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return img

@app.post("/predict", response_model=Resp)
async def predict(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.5):
    img0 = bytes_to_bgr(await file.read())
    h0, w0 = img0.shape[:2]
    img, r, pad = letterbox(img0, IMGSZ)
    x = torch.from_numpy(bgr_to_tensor(img)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        boxes, scores, cls_ids = decode_model_output(pred, conf_thres=conf)
        boxes, scores, cls_ids = multiclass_nms(boxes, scores, cls_ids, iou_thres=iou, conf_thres=conf)
        boxes = scale_boxes_back(boxes, r, pad, (h0, w0))

    dets: list[Det] = []
    for b, s, c in zip(boxes.cpu().numpy(), scores.cpu().numpy(), cls_ids.cpu().numpy()):
        c = int(c)
        dets.append(Det(cls=c, label=NAMES.get(c, str(c)), conf=float(s), xyxy=[float(x) for x in b.tolist()]))

    return Resp(detections=dets)