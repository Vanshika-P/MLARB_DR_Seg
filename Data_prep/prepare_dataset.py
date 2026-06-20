#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# =========================================================
# CHANGE LESION TYPE HERE
# =========================================================

LESION_TYPE = "MA"  # change for every lesion

# options:
# "MA"
# "HE"
# "EX"
# "SE"

# =========================================================
# LOAD IMAGE
# =========================================================

def load_rgb(path: Path):

    return np.array(
        Image.open(path).convert("RGB"),
        dtype=np.uint8
    )

# =========================================================
# LOAD MASK
# =========================================================

def load_mask_to_ids(path: Path):

    mask = np.array(Image.open(path))

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask = (mask > 0).astype(np.uint8)

    return mask

# =========================================================
# SAVE PNG
# =========================================================

def save_png(path: Path, arr: np.ndarray):

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    Image.fromarray(arr).save(path)

# =========================================================
# ROI FROM MASK
# =========================================================

def crop_roi_from_mask(label, pad=64):

    ys, xs = np.where(label != 0)

    if ys.size == 0:
        return 0, label.shape[0], 0, label.shape[1]

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - pad)
    y1 = min(label.shape[0], y1 + pad + 1)

    x0 = max(0, x0 - pad)
    x1 = min(label.shape[1], x1 + pad + 1)

    return y0, y1, x0, x1

# =========================================================
# TILE COORDS
# =========================================================

def tile_coords(h, w, patch, overlap):

    step = patch - overlap

    ys = list(range(0, max(1, h - patch + 1), step))
    xs = list(range(0, max(1, w - patch + 1), step))

    if ys[-1] != h - patch:
        ys.append(h - patch)

    if xs[-1] != w - patch:
        xs.append(w - patch)

    for y in ys:
        for x in xs:
            yield y, x

# =========================================================
# FOREGROUND RATIO
# =========================================================

def non_bg_ratio(lbl):

    total = lbl.size

    if total == 0:
        return 0.0

    return np.count_nonzero(lbl) / float(total)

# =========================================================
# COLLECT IMAGE-MASK PAIRS
# =========================================================

def collect_pairs(folder: Path):

    pairs = []

    img_paths = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        and "_MA" not in p.stem
        and "_HE" not in p.stem
        and "_EX" not in p.stem
        and "_SE" not in p.stem
    ])

    for img_path in img_paths:

        img_id = img_path.stem

        mask_path = None

        possible_masks = [
            folder / f"{img_id}_{LESION_TYPE}.tif",
            folder / f"{img_id}_{LESION_TYPE}.tiff",
            folder / f"{img_id}_{LESION_TYPE}.png",
        ]

        for m in possible_masks:

            if m.exists():
                mask_path = m
                break

        if mask_path is not None:
            pairs.append((img_path, mask_path))

    return pairs

# =========================================================
# PROCESS SPLIT
# =========================================================

def process_split(
    split,
    in_dir,
    out_img_dir,
    out_msk_dir,
    manifest_csv,
    patch,
    overlap,
    min_fg,
    pad,
):

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(in_dir)

    print(f"\nFound {len(pairs)} image-mask pairs in {split}\n")

    kept_tiles = 0
    skipped_tiles = 0

    with open(manifest_csv, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "image_path",
            "mask_path",
            "src_id",
            "h",
            "w",
            "y",
            "x",
        ])

        for img_path, mask_path in tqdm(pairs, desc=f"{split}: files"):

            src_id = img_path.stem

            img = load_rgb(img_path)

            label = load_mask_to_ids(mask_path)

            if label.shape[:2] != img.shape[:2]:

                label = np.array(
                    Image.fromarray(label).resize(
                        (img.shape[1], img.shape[0]),
                        resample=Image.NEAREST,
                    ),
                    dtype=np.uint8,
                )

            y0, y1, x0, x1 = crop_roi_from_mask(label, pad=pad)

            img_c = img[y0:y1, x0:x1]
            lbl_c = label[y0:y1, x0:x1]

            H, W = img_c.shape[:2]

            if H < patch or W < patch:

                py = max(0, patch - H)
                px = max(0, patch - W)

                img_c = np.pad(
                    img_c,
                    ((0, py), (0, px), (0, 0)),
                    mode="constant",
                    constant_values=255,
                )

                lbl_c = np.pad(
                    lbl_c,
                    ((0, py), (0, px)),
                    mode="constant",
                    constant_values=0,
                )

                H, W = img_c.shape[:2]

            for y, x in tile_coords(H, W, patch, overlap):

                img_t = img_c[y:y+patch, x:x+patch]
                lbl_t = lbl_c[y:y+patch, x:x+patch]

                # =====================================================
                # IMPORTANT FILTER
                # removes almost-empty patches
                # helps prevent background collapse
                # =====================================================

                if non_bg_ratio(lbl_t) < min_fg:
                    skipped_tiles += 1
                    continue

                kept_tiles += 1

                img_name = f"img_{src_id}_{y0+y}_{x0+x}.png"

                msk_name = f"mask_{src_id}_{y0+y}_{x0+x}.png"

                save_png(out_img_dir / img_name, img_t)

                save_png(out_msk_dir / msk_name, lbl_t)

                writer.writerow([
                    str(out_img_dir / img_name),
                    str(out_msk_dir / msk_name),
                    src_id,
                    patch,
                    patch,
                    y0+y,
                    x0+x,
                ])

    print(f"\n[{split}] kept patches   : {kept_tiles}")
    print(f"[{split}] skipped patches: {skipped_tiles}")

# =========================================================
# COMPUTE STATS
# =========================================================

def compute_train_mean_std(train_images_dir):

    paths = list(train_images_dir.glob("*.png"))

    if not paths:
        return [0.5]*3, [0.25]*3

    n_pixels = 0

    sum_rgb = np.zeros(3, dtype=np.float64)

    sumsq_rgb = np.zeros(3, dtype=np.float64)

    for p in tqdm(paths, desc="compute mean/std"):

        im = (
            np.array(
                Image.open(p).convert("RGB"),
                dtype=np.float32
            ) / 255.0
        )

        h, w, _ = im.shape

        n = h * w

        n_pixels += n

        sum_rgb += im.reshape(-1, 3).sum(axis=0)

        sumsq_rgb += (im.reshape(-1, 3) ** 2).sum(axis=0)

    mean = (sum_rgb / n_pixels).tolist()

    var = (sumsq_rgb / n_pixels) - np.square(mean)

    std = np.sqrt(np.maximum(var, 1e-12)).tolist()

    return mean, std

# =========================================================
# MAIN
# =========================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, required=True)

    ap.add_argument(
        "--out-name",
        type=str,
        default=f"idrid_{LESION_TYPE.lower()}",
    )

    ap.add_argument("--patch-size", type=int, default=256)

    ap.add_argument("--overlap", type=int, default=64)

    # =====================================================
    # IMPORTANT:
    # filters almost-empty patches
    # 0.0005 ≈ 32 positive pixels in 256x256 patch
    # =====================================================

    ap.add_argument(
        "--min-foreground",
        type=float,
        default=0.0005
    )

    ap.add_argument("--pad", type=int, default=64)

    args = ap.parse_args()

    root = Path(args.root).resolve()

    raw = root / "raw"

    processed = root / "processed" / args.out_name

    manifests = root / "processed" / "manifests"

    manifests.mkdir(parents=True, exist_ok=True)

    train_in = raw / "training"

    val_in = raw / "validation"

    train_img_out = processed / "train" / "images"

    train_msk_out = processed / "train" / "masks"

    val_img_out = processed / "val" / "images"

    val_msk_out = processed / "val" / "masks"

    # =====================================================
    # LESION-SPECIFIC MANIFESTS
    # =====================================================

    train_csv = manifests / f"train_{LESION_TYPE.lower()}.csv"

    val_csv = manifests / f"val_{LESION_TYPE.lower()}.csv"

    process_split(
        "train",
        train_in,
        train_img_out,
        train_msk_out,
        train_csv,
        patch=args.patch_size,
        overlap=args.overlap,
        min_fg=args.min_foreground,
        pad=args.pad,
    )

    process_split(
        "val",
        val_in,
        val_img_out,
        val_msk_out,
        val_csv,
        patch=args.patch_size,
        overlap=args.overlap,
        min_fg=args.min_foreground,
        pad=args.pad,
    )

    mean, std = compute_train_mean_std(train_img_out)

    stats_path = root / "processed" / "stats.json"

    stats = {
        "rgb_mean": mean,
        "rgb_std": std,
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone preprocessing for {LESION_TYPE}")

# =========================================================

if __name__ == "__main__":
    main()