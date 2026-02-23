from dataclasses import dataclass

@dataclass
class InferConfig:
    weights: str = "weights/best.pt"
    source: str = "images"      # file/dir/video/0(webcam)
    imgsz: int = 960
    conf: float = 0.25
    iou: float = 0.5
    device: str = "cuda"        # or "cpu"
    half: bool = True
    out_dir: str = "outputs"
    class_names: dict[int, str] | None = None  # {0:"crack",1:"corrosion",...}

@dataclass
class TrainConfig:
    weights: str | None = None  # optional resume/init
    data_root: str = "dataset"
    imgsz: int = 960
    batch: int = 8
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 5e-4
    device: str = "cuda"
    num_workers: int = 4
    save_dir: str = "runs/exp"