from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import torch

from utils.preproc import letterbox, bgr_to_tensor
from utils.nms import multiclass_nms
from utils.postproc import scale_boxes_back, draw_detections
from predict import load_model, decode_model_output

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", default="0", help="0 webcam or video path")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--half", action="store_true")
    p.add_argument("--out", default="outputs/video.mp4")
    p.add_argument("--names", default="")
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

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video/webcam source")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        img, r, pad = letterbox(frame, new_shape=args.imgsz)
        x = torch.from_numpy(bgr_to_tensor(img)).unsqueeze(0).to(device)
        if args.half and device.startswith("cuda"):
            x = x.half()

        with torch.no_grad():
            pred = model(x)
            boxes, scores, cls_ids = decode_model_output(pred, conf_thres=args.conf)
            boxes, scores, cls_ids = multiclass_nms(boxes, scores, cls_ids, args.iou, args.conf)
            boxes = scale_boxes_back(boxes, r, pad, (frame.shape[0], frame.shape[1]))

        vis = draw_detections(frame, boxes.cpu().numpy(), scores.cpu().numpy(), cls_ids.cpu().numpy(), names=names)
        writer.write(vis)

    cap.release()
    writer.release()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()