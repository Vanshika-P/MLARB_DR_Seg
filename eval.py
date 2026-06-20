import os, csv, json, argparse
from collections import defaultdict
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# =========================================================
# LESION COLORS
# =========================================================

LESION_COLORS = {
    "ma": [255, 0, 0],      # red
    "he": [0, 255, 0],      # green
    "ex": [255, 255, 0],    # yellow
    "se": [0, 0, 255],      # blue
}

# =========================================================
# IO HELPERS
# =========================================================

def read_manifest(csv_path):

    rows = []

    with open(csv_path, newline="") as f:

        r = csv.DictReader(f)

        for row in r:

            row["h"] = int(row["h"])
            row["w"] = int(row["w"])
            row["y"] = int(row["y"])
            row["x"] = int(row["x"])

            rows.append(row)

    return rows


def load_rgb(path):

    return Image.open(path).convert("RGB")


def load_mask_ids(path):

    arr = np.array(Image.open(path), dtype=np.uint8)

    arr = (arr > 0).astype(np.uint8)

    return arr


def make_val_transform(img_size, mean, std):

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(mean, std),
    ])


def upsample_to(arr_chw, out_h, out_w, mode="bilinear"):

    t = arr_chw.unsqueeze(0)

    t = F.interpolate(
        t,
        size=(out_h, out_w),
        mode=mode,
        align_corners=False if mode == "bilinear" else None
    )

    return t.squeeze(0)

# =========================================================
# METRICS
# =========================================================

def iou_binary(pred_ids, gt_ids):

    pred_c = (pred_ids == 1)
    gt_c = (gt_ids == 1)

    inter = np.logical_and(pred_c, gt_c).sum()
    union = np.logical_or(pred_c, gt_c).sum()

    return (inter / union) if union > 0 else 0.0


def dice_binary(pred_ids, gt_ids):

    pred_c = (pred_ids == 1)
    gt_c = (gt_ids == 1)

    inter = np.logical_and(pred_c, gt_c).sum()
    denom = pred_c.sum() + gt_c.sum()

    return (2 * inter / denom) if denom > 0 else 0.0


def precision_recall_binary(pred_ids, gt_ids):

    pred_c = (pred_ids == 1)
    gt_c = (gt_ids == 1)

    tp = np.logical_and(pred_c, gt_c).sum()
    fp = np.logical_and(pred_c, ~gt_c).sum()
    fn = np.logical_and(~pred_c, gt_c).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall

def specificity_binary(pred_ids, gt_ids):

    pred_c = (pred_ids == 1)
    gt_c = (gt_ids == 1)

    tn = np.logical_and(~pred_c, ~gt_c).sum()
    fp = np.logical_and(pred_c, ~gt_c).sum()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return specificity


# =========================================================
# MAIN EVAL
# =========================================================

@torch.no_grad()
def stitch_and_eval(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------

    stats_path = os.path.join(
        args.data_root,
        "processed",
        "stats.json"
    )

    with open(stats_path) as f:

        stats = json.load(f)

    mean = stats["rgb_mean"]
    std = stats["rgb_std"]

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    from Models import models as modelz

    model = modelz.FCBFormer(
        size=args.img_size,
        num_classes=1
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)

    state = ckpt.get("model_state_dict", ckpt)

    model.load_state_dict(state)

    model.eval()

    # -----------------------------------------------------
    # LESION-SPECIFIC MANIFEST
    # -----------------------------------------------------

    val_csv = os.path.join(
        args.data_root,
        "processed",
        "manifests",
        f"val_{args.lesion}.csv"
    )

    rows = read_manifest(val_csv)

    by_src = defaultdict(list)

    for r in rows:


        by_src[r["src_id"]].append(r)

    tf = make_val_transform(args.img_size, mean, std)

    lesion_out_dir = os.path.join(args.out_dir, args.lesion.upper())

    os.makedirs(lesion_out_dir, exist_ok=True)

    all_iou = []
    all_dice = []
    all_precision = []
    all_recall = []
    all_specificity = []

    # =====================================================

    for idx, (src_id, tiles) in enumerate(sorted(by_src.items())):

        H = max(r["y"] + r["h"] for r in tiles)
        W = max(r["x"] + r["w"] for r in tiles)

        logit_acc = torch.zeros(
            1,
            H,
            W,
            dtype=torch.float32,
            device=device
        )

        count_acc = torch.zeros(
            1,
            H,
            W,
            dtype=torch.float32,
            device=device
        )

        gt_stitched = np.zeros((H, W), dtype=np.uint8)

        batch_imgs = []
        batch_meta = []

        BATCH = args.batch_size

        # -------------------------------------------------

        def flush_batch():

            if not batch_imgs:
                return

            x = torch.stack(batch_imgs, dim=0).to(device)

            logits = model(x)

            if isinstance(logits, (tuple, list)):

                logits = logits[0]

            for b in range(logits.shape[0]):

                r = batch_meta[b]

                logit_tile = logits[b]

                logit_tile = upsample_to(
                    logit_tile,
                    r["h"],
                    r["w"],
                    mode="bilinear"
                )

                y0, x0 = r["y"], r["x"]

                logit_acc[:, y0:y0+r["h"], x0:x0+r["w"]] += logit_tile

                count_acc[:, y0:y0+r["h"], x0:x0+r["w"]] += 1.0

            batch_imgs.clear()
            batch_meta.clear()

        # -------------------------------------------------

        for r in tiles:

            img = load_rgb(r["image_path"])

            gt = load_mask_ids(r["mask_path"])

            gt_stitched[
                r["y"]:r["y"]+r["h"],
                r["x"]:r["x"]+r["w"]
            ] = gt

            x = tf(img)

            batch_imgs.append(x)
            batch_meta.append(r)

            if len(batch_imgs) == BATCH:

                flush_batch()

        flush_batch()

        # -------------------------------------------------

        count_acc = torch.clamp(count_acc, min=1.0)

        logit_acc /= count_acc

        pred_probs = torch.sigmoid(logit_acc).squeeze(0)

        pred_ids = (
            pred_probs > 0.5
        ).cpu().numpy().astype(np.uint8)

        # -------------------------------------------------

        iou = iou_binary(pred_ids, gt_stitched)

        dice = dice_binary(pred_ids, gt_stitched)

        precision, recall = precision_recall_binary(
            pred_ids,
            gt_stitched
        )

        specificity = specificity_binary(
            pred_ids,
            gt_stitched
        )
        all_iou.append(iou)
        all_dice.append(dice)
        all_precision.append(precision)
        all_recall.append(recall)
        all_specificity.append(specificity)

        # -------------------------------------------------

        if args.save_png:

            color_map = np.array([
                [0, 0, 0],
                LESION_COLORS[args.lesion]
            ], dtype=np.uint8)

            color = color_map[pred_ids]

            Image.fromarray(color).save(
                os.path.join(
                    lesion_out_dir,
                    f"{src_id}_pred.png"
                )
            )

        # -------------------------------------------------

        print(f"[{idx+1}/{len(by_src)}] {src_id}")
        print(f"  IoU         : {iou:.4f}")
        print(f"  Dice        : {dice:.4f}")
        print(f"  Precision   : {precision:.4f}")
        print(f"  Recall      : {recall:.4f}")
        print(f"  Specificity : {specificity:.4f}")

    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print("\n==============================")
    print("FINAL VALIDATION RESULTS")
    print("==============================")

    print(f"Mean IoU         : {np.mean(all_iou):.4f}")
    print(f"Mean Dice        : {np.mean(all_dice):.4f}")
    print(f"Mean Precision   : {np.mean(all_precision):.4f}")
    print(f"Mean Recall      : {np.mean(all_recall):.4f}")
    print(f"Mean Specificity : {np.mean(all_specificity):.4f}")

# =========================================================
# ARGS
# =========================================================

def parse_args():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", required=True)

    ap.add_argument("--checkpoint", required=True)

    ap.add_argument(
        "--lesion",
        type=str,
        required=True,
        choices=["ma", "he", "ex", "se"]
    )

    ap.add_argument("--img-size", type=int, default=256)

    ap.add_argument("--batch-size", type=int, default=4)

    ap.add_argument("--out-dir", type=str, default="./EvalIDRiD")

    ap.add_argument("--save-png", action="store_true")

    return ap.parse_args()

# =========================================================
# MAIN
# =========================================================

def main():

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    args = parse_args()

    stitch_and_eval(args)

# =========================================================

if __name__ == "__main__":
    main()