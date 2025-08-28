#!/usr/bin/env python3
import argparse
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import torch
import cv2

# optional GeoTIFF IO
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except Exception:
    RASTERIO_AVAILABLE = False

from oem_lightweight.model import sparsemask, fasterseg  # repo loaders


# -------------------------
# logging setup
# -------------------------
def setup_logging(level: str, log_file: str | None):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)5s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)5s | %(message)s"))
        logging.getLogger().addHandler(fh)


def log_env():
    logging.info("Python %s", sys.version.split()[0])
    logging.info("PyTorch %s  CUDA available: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("CUDA device: %s", torch.cuda.get_device_name(0))


# -------------------------
# compat_demo helpers
# -------------------------
def load_image_opencv_bgr(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    H, W, _ = im.shape
    im = im.astype("float32") / 255.0
    return im, H, W


def resize_and_pad_1024(img_bgr01, size=1024):
    H, W, _ = img_bgr01.shape
    scale = min(size / H, size / W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    resized = cv2.resize(img_bgr01, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad = np.zeros((size, size, 3), dtype=np.float32)
    pad[:new_h, :new_w, :] = resized
    return pad, new_h, new_w


def apply_eval_preprocessing(img_bgr01, model_name):
    """Apply the exact preprocessing used in eval_oem_lightweight.py"""
    # Convert to uint8 like the evaluator does
    img_uint8 = (img_bgr01 * 255).astype(np.uint8)
    
    # For FasterSeg, flip BGR to RGB - make a copy to avoid negative strides
    if model_name.lower() == "fasterseg":
        img_uint8 = img_uint8[:, :, ::-1].copy()
    
    # Use PyTorch transforms like the evaluator - convert to tensor first then normalize
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4325, 0.4483, 0.3879], std=[0.0195, 0.0169, 0.0179])
    ])
    
    # transforms expect PIL Image or numpy array HWC
    img_tensor = transform(img_uint8)  # This gives CHW tensor
    return img_tensor


def to_tensor(img_hw3):
    return torch.from_numpy(np.transpose(img_hw3, (2, 0, 1))).unsqueeze(0)


def forward_logits_strict(model, x):
    y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    assert y.ndim == 4 and y.shape[1] >= 2, f"Unexpected output shape: {tuple(y.shape)}"
    return y


def tta(model, input_tensor):
    """Test Time Augmentation - same as evaluator.py"""
    score = model(input_tensor)
    input_flip = input_tensor.flip(-1)
    score_flip = model(input_flip)
    score += score_flip.flip(-1)
    score = torch.exp(score)
    return score


# -------------------------
# generic preprocessing (non-compat)
# -------------------------
def load_image(path):
    if RASTERIO_AVAILABLE:
        with rasterio.open(path) as src:
            arr = src.read(out_dtype="float32")  # CxHxW
            meta = src.meta.copy()
        if arr.shape[0] >= 3:
            arr = arr[:3]
        else:
            arr = np.repeat(arr[:1], 3, axis=0)
        img = np.transpose(arr, (1, 2, 0))
        return img, meta
    else:
        from PIL import Image
        pil = Image.open(path).convert("RGB")
        img = np.asarray(pil).astype("float32")
        return img, None


def auto_range_scale(img):
    vmax = float(img.max())
    if vmax > 1.5:
        return np.clip(img, 0, 255) / 255.0
    else:
        return np.clip(img, 0, 1)


# -------------------------
# model + inference
# -------------------------
def build_model(model_name, arch_path, weights_path, device):
    if model_name.lower() == "sparsemask":
        m = sparsemask(mask=arch_path, weights=weights_path)
    elif model_name.lower() == "fasterseg":
        m = fasterseg(arch=arch_path, weights=weights_path)
    net = m["model"].to(device).eval()
    return net


@torch.inference_mode()
def run_inference_tiled(model, img_hw3, device="cuda", tile=1024, stride=None):
    H, W, _ = img_hw3.shape
    if stride is None:
        stride = tile // 2
    dummy = torch.zeros(1, 3, tile, tile, device=device)
    C = forward_logits_strict(model, dummy).shape[1]
    logit_accum = torch.zeros(C, H, W, device=device)
    weight = torch.zeros(1, H, W, device=device)

    tiles = 0
    for top in range(0, H, stride):
        if top + tile > H:
            top = H - tile
        for left in range(0, W, stride):
            if left + tile > W:
                left = W - tile
            patch = img_hw3[top:top+tile, left:left+tile, :]
            if patch.shape[0] != tile or patch.shape[1] != tile:
                pad_h = tile - patch.shape[0]
                pad_w = tile - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            x = to_tensor(patch).to(device)
            logits = forward_logits_strict(model, x)[0]
            h = min(tile, H - top)
            w = min(tile, W - left)
            logit_accum[:, top:top+h, left:left+w] += logits[:, :h, :w]
            weight[:, top:top+h, left:left+w] += 1
            tiles += 1
            if left + tile >= W:
                break
        if top + tile >= H:
            break

    weight = torch.clamp(weight, min=1.0)
    logit_accum /= weight
    pred = torch.argmax(logit_accum, dim=0).cpu().numpy().astype(np.uint8)
    return pred, C


# -------------------------
# viz + saving
# -------------------------
def colorize(mask_hw):
    # Use the exact same colors as config.py
    colors = np.array([
        [128, 0, 0],    # 0: bareland
        [0, 255, 0],    # 1: rangeland  
        [192, 192, 192], # 2: developed space
        [255, 255, 255], # 3: road
        [49, 139, 87],   # 4: tree
        [0, 0, 255],     # 5: water
        [127, 255, 0],   # 6: agriculture land
        [255, 0, 0],     # 7: buildings
    ], dtype=np.uint8)
    mask_hw = np.clip(mask_hw, 0, len(colors)-1)
    return colors[mask_hw]


def save_outputs(pred_ids, out_stem, color=True, geotiff_meta=None):
    from PIL import Image
    Image.fromarray(pred_ids, mode="L").save(f"{out_stem}_pred_ids.png")
    if color:
        Image.fromarray(colorize(pred_ids), mode="RGB").save(f"{out_stem}_pred_color.png")
    if geotiff_meta is not None and RASTERIO_AVAILABLE:
        meta = geotiff_meta.copy()
        meta.update({"count": 1, "dtype": "uint8"})
        meta.pop("photometric", None)
        with rasterio.open(f"{out_stem}_pred_ids.tif", "w", **meta) as dst:
            dst.write(pred_ids[np.newaxis, ...].astype("uint8"))


def summarize_pred(pred, C):
    vals, counts = np.unique(pred, return_counts=True)
    total = pred.size
    logging.info("Prediction class histogram (C=%d):", C)
    for v, c in zip(vals, counts):
        logging.info("  class %d : %d (%.2f%%)", v, c, 100*c/total)
    if len(vals) == 1:
        logging.warning("Degenerate prediction: only class %d present", vals[0])


# -------------------------
# main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="OEM lightweight inference (compat or tiled)")
    p.add_argument("--model", required=True, choices=["fasterseg", "sparsemask"])
    p.add_argument("--arch", required=True)
    p.add_argument("--pretrained_weights", required=True)
    p.add_argument("--image_file", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--compat_demo", action="store_true", help="Mimic repo demo pipeline (OpenCV BGR, resize+pad 1024)")
    p.add_argument("--use_tta", action="store_true", help="Use Test Time Augmentation (like evaluator)")
    p.add_argument("--out", default=None)
    p.add_argument("--no_color", action="store_true")
    p.add_argument("--loglevel", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = p.parse_args()

    setup_logging(args.loglevel, None)
    log_env()

    device = args.device
    net = build_model(args.model, args.arch, args.pretrained_weights, device)

    if args.compat_demo:
        logging.info("Running in compat_demo mode")
        raw_bgr01, orig_h, orig_w = load_image_opencv_bgr(args.image_file)
        padded, new_h, new_w = resize_and_pad_1024(raw_bgr01, size=1024)
        
        # Apply the exact same preprocessing as evaluator (returns CHW tensor)
        processed_tensor = apply_eval_preprocessing(padded, args.model)
        x = processed_tensor.unsqueeze(0).to(device)  # Add batch dimension: NCHW
        
        # Use TTA like the evaluator does
        with torch.no_grad():
            if args.use_tta:
                logits = tta(net, x)
            else:
                logits = net(x)
        
        C = logits.shape[1]
        logits = logits[:, :, :new_h, :new_w]
        pred_resized = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_ids = cv2.resize(pred_resized, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        num_classes = C
    else:
        logging.info("Running in generic tiling mode")
        img, meta = load_image(args.image_file)
        img01 = auto_range_scale(img)
        pred_ids, num_classes = run_inference_tiled(net, img01, device=device, tile=args.tile, stride=args.stride)

    summarize_pred(pred_ids, num_classes)
    if args.out is None:
        stem = Path(args.image_file).with_suffix("").name
        out_stem = str(Path("results") / stem)
    else:
        out_stem = args.out
    os.makedirs(Path(out_stem).parent, exist_ok=True)

    save_outputs(pred_ids, out_stem, color=(not args.no_color), geotiff_meta=None)
    logging.info("Saved results to: %s_pred_ids.png", out_stem)
    if not args.no_color:
        logging.info("Saved results to: %s_pred_color.png", out_stem)


if __name__ == "__main__":
    main()
