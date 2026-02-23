from __future__ import annotations
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model.model import ObjNano
from data.yolo_dataset import YoloTxtDetectionDataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="dataset")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights", default="", help="optional init weights")
    p.add_argument("--save_dir", default="runs/exp")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_classes", type=int, default=80, help="number of classes")
    return p.parse_args()

def load_weights(model, weights: str):
    if not weights:
        return
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    clean = {k.replace("module.", ""): v for k, v in state.items()} if isinstance(state, dict) else state
    model.load_state_dict(clean, strict=False)

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    model = ObjNano(num_classes=args.num_classes)
    load_weights(model, args.weights)
    model.to(device).train()

    ds = YoloTxtDetectionDataset(args.data_root, split="train", imgsz=args.imgsz)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=lambda b: b)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0.0
        n_batches = 0

        for batch in pbar:
            # batch is list of (x,y) because of collate_fn
            xs = torch.stack([item[0] for item in batch], dim=0).to(device)
            ys = [item[1].to(device) for item in batch]  # variable length targets per image

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                pred = model(xs)

                # ---- REQUIRED: you must provide this in your model OR replace this block ----
                if hasattr(model, "compute_loss"):
                    loss = model.compute_loss(pred, ys)
                else:
                    raise RuntimeError(
                        "Your model.py must define model.compute_loss(pred, targets) OR you must plug in your YOLOv8 loss here."
                    )

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=total_loss / max(1, n_batches))

        # save checkpoint
        ckpt_path = save_dir / f"epoch_{epoch+1}.pt"
        torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, ckpt_path)

    print(f"Done. Saved to {save_dir}")

if __name__ == "__main__":
    main()