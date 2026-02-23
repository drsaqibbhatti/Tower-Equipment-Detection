from __future__ import annotations
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def iter_images(source: str):
    p = Path(source)
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        yield str(p)
        return
    if p.is_dir():
        for fp in sorted(p.rglob("*")):
            if fp.suffix.lower() in IMG_EXTS:
                yield str(fp)
        return
    raise ValueError(f"Unsupported image source: {source}")