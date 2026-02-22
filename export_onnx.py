from __future__ import annotations
import argparse
import torch
from model.model import ObjNano

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--out", default="model.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--num_classes", type=int, default=80, help="number of classes")
    return p.parse_args()

def main():
    args = parse_args()
    model = ObjNano(num_classes=args.num_classes)
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    model.eval()

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        opset_version=args.opset,
        input_names=["images"],
        output_names=["pred"],
        dynamic_axes={"images": {0: "batch"}, "pred": {0: "batch"}},
    )
    print(f"Exported: {args.out}")

if __name__ == "__main__":
    main()