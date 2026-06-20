import sys
import os
import argparse
import time
import numpy as np
import glob
from pathlib import Path
from datetime import datetime
import subprocess
import json
import random

import torch
import torch.nn as nn

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
from Metrics.log_helper import AvgMeter

from Data.dataloaders_joint import get_loaders_from_manifests


def train_epoch(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    Dice_loss,
    BCE_loss,
    ce_loss=None,
    dice_loss_mc=None,
    bnd_loss=None,
    tumor_dil_loss=None,
    benign_dil_loss=None,
    boundary_weight=0.2,
    tumor_dil_weight=0.2,
    benign_dil_weight=0.2,
    num_classes=1,
):

    t = time.time()

    model.train()

    loss_accumulator = []

    m_ce = AvgMeter()
    m_dice = AvgMeter()
    m_bnd = AvgMeter()
    m_tum = AvgMeter()
    m_ben = AvgMeter()

    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)

        # =========================================================
        # MULTICLASS (AIRA)
        # =========================================================

        if num_classes > 1:

            target_long = target.long()

            l_dice = dice_loss_mc(output, target_long)
            m_dice.update(l_dice.item(), data.size(0))

            l_ce = ce_loss(output, target_long)
            m_ce.update(l_ce.item(), data.size(0))

            loss = l_dice + l_ce

            if bnd_loss is not None and boundary_weight > 0:

                l_bnd = bnd_loss(output, target_long)

                m_bnd.update(l_bnd.item(), data.size(0))

                loss = loss + boundary_weight * l_bnd

            if tumor_dil_loss is not None and tumor_dil_weight > 0:

                l_tum = tumor_dil_loss(output, target_long)

                m_tum.update(l_tum.item(), data.size(0))

                loss = loss + tumor_dil_weight * l_tum

            if benign_dil_loss is not None and benign_dil_weight > 0:

                l_ben = benign_dil_loss(output, target_long)

                m_ben.update(l_ben.item(), data.size(0))

                loss = loss + benign_dil_weight * l_ben

        # =========================================================
        # BINARY (IDRiD)
        # =========================================================

        else:

            target = target.unsqueeze(1).float()

            l_dice = Dice_loss(output, target)
            m_dice.update(l_dice.item(), data.size(0))

            l_ce = BCE_loss(output, target)
            m_ce.update(l_ce.item(), data.size(0))

            loss = l_dice + l_ce

        loss.backward()

        optimizer.step()

        loss_accumulator.append(loss.item())

        if batch_idx + 1 < len(train_loader):

            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )

        else:

            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )

    return {
        "loss": float(np.mean(loss_accumulator)),
        "ce": m_ce.avg,
        "dice": m_dice.avg,
        "bnd": m_bnd.avg,
        "tum": m_tum.avg,
        "ben": m_ben.avg,
    }


@torch.no_grad()
def iou_per_class(logits, targets, num_classes=4):

    preds = torch.argmax(logits, dim=1)

    ious = []

    for c in range(num_classes):

        pred_c = preds == c
        targ_c = targets == c

        inter = (pred_c & targ_c).sum().item()
        union = (pred_c | targ_c).sum().item()

        ious.append(inter / union if union > 0 else 0.0)

    return ious, sum(ious) / len(ious)


@torch.no_grad()
def test(model, device, test_loader, epoch, perf_measure=None, num_classes=4):

    model.eval()

    perf_accumulator = []
    all_ious = []

    for batch_idx, (data, target) in enumerate(test_loader):

        data = data.to(device)
        target = target.to(device)

        output = model(data)

        if num_classes == 1:

            target = target.unsqueeze(1).float()

        if perf_measure is not None and num_classes == 1:

            perf_accumulator.append(
                perf_measure(output, target).item()
            )

        if num_classes > 1:

            ious, miou = iou_per_class(
                output,
                target,
                num_classes=num_classes,
            )

            all_ious.append(ious)

    if num_classes > 1 and all_ious:

        mean_ious = np.mean(all_ious, axis=0).tolist()

        miou = float(np.mean(mean_ious))

        print(f"[Val IoU] per-class: {mean_ious}, mIoU: {miou:.4f}")

        return (
            miou,
            0.0,
            {
                "bg": mean_ious[0],
                "stroma": mean_ious[1],
                "benign": mean_ious[2],
                "tumor": mean_ious[3],
            },
        )

    return (
        float(np.mean(perf_accumulator)) if perf_accumulator else 0.0,
        float(np.std(perf_accumulator)) if perf_accumulator else 0.0,
        {},
    )


def build(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(device)

    # =========================================================
    # Kvasir / CVC
    # =========================================================

    if args.dataset in ["Kvasir", "CVC"]:

        if args.dataset == "Kvasir":

            img_path = args.root + "images/*"
            input_paths = sorted(glob.glob(img_path))

            depth_path = args.root + "masks/*"
            target_paths = sorted(glob.glob(depth_path))

        else:

            img_path = args.root + "Original/*"
            input_paths = sorted(glob.glob(img_path))

            depth_path = args.root + "Ground Truth/*"
            target_paths = sorted(glob.glob(depth_path))

        train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
            input_paths,
            target_paths,
            batch_size=args.batch_size,
        )

        num_classes = 1

        ce_loss = nn.BCEWithLogitsLoss()

        dice_loss = losses.SoftDiceLoss()

        bnd_loss = None
        tumor_dil_loss = None
        benign_dil_loss = None

    # =========================================================
    # AIRA / IDRiD
    # =========================================================

    elif args.dataset in ["AIRA", "IDRiD"]:

        train_csv = os.path.join(
            args.root,
            "processed",
            "manifests",
            f"train_{args.lesion}.csv",
        )

        val_csv = os.path.join(
            args.root,
            "processed",
            "manifests",
            f"val_{args.lesion}.csv",
        )

        stats_json = os.path.join(
            args.root,
            "processed",
            "stats.json",
        )

        train_dataloader, val_dataloader = get_loaders_from_manifests(
            train_csv,
            val_csv,
            stats_json,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=0,
        )

        # =====================================================
        # IDRiD Binary
        # =====================================================

        if args.dataset == "IDRiD":

            num_classes = 1

            ce_loss = nn.BCEWithLogitsLoss()

            dice_loss = losses.SoftDiceLoss()

            bnd_loss = None
            tumor_dil_loss = None
            benign_dil_loss = None

        # =====================================================
        # AIRA Multiclass
        # =====================================================

        else:

            num_classes = 4

            with open(stats_json) as f:
                stats = json.load(f)

            cc = stats["class_counts"]

            freq = torch.tensor(
                [
                    cc["background"],
                    cc["stroma"],
                    cc["benign"],
                    cc["tumor"],
                ],
                dtype=torch.float,
            )

            weights = 1.0 / torch.log(1.02 + freq / freq.sum())

            weights = weights / weights.mean()

            weights = weights.to(device)

            ce_loss = nn.CrossEntropyLoss(weight=weights)

            dice_loss = losses.MultiClassDiceLoss(
                num_classes=num_classes
            )

            bnd_loss = losses.BoundaryDiceLossAgnostic(
                num_classes=num_classes,
                kernel_size=3,
            )

            tumor_dil_loss = losses.TumorDilatedDiceLoss(
                num_classes=num_classes,
                iters=args.dil_iters,
                kernel_size=args.dil_kernel,
                prob_power=args.prob_power,
            )

            benign_dil_loss = losses.BenignDilatedDiceLoss(
                num_classes=num_classes,
                iters=args.dil_iters,
                kernel_size=args.dil_kernel,
                prob_power=args.prob_power,
            )

    else:

        raise ValueError("Unknown dataset")

    model = models.FCBFormer(
        size=args.img_size,
        num_classes=num_classes,
    )

    if args.mgpu == "true":

        model = nn.DataParallel(model)

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    perf = (
        performance_metrics.DiceScore()
        if num_classes == 1
        else None
    )

    return (
        device,
        train_dataloader,
        val_dataloader,
        dice_loss,
        ce_loss,
        bnd_loss,
        tumor_dil_loss,
        benign_dil_loss,
        perf,
        model,
        optimizer,
        num_classes,
    )


def train(args, run_dir: Path):

    (
        device,
        train_dataloader,
        val_dataloader,
        Dice_loss,
        CE_loss,
        BND_loss,
        TumorDilLoss,
        BenignDilLoss,
        perf,
        model,
        optimizer,
        num_classes,
    ) = build(args)

    prev_best_test = None

    scheduler = None

    if args.lrs == "true":

        if args.lrs_min > 0:

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                min_lr=args.lrs_min,
            )

        else:

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
            )

    for epoch in range(1, args.epochs + 1):

        train_stats = train_epoch(
            model,
            device,
            train_dataloader,
            optimizer,
            epoch,
            Dice_loss,
            CE_loss,
            ce_loss=CE_loss if num_classes > 1 else None,
            dice_loss_mc=Dice_loss if num_classes > 1 else None,
            bnd_loss=BND_loss if num_classes > 1 else None,
            tumor_dil_loss=TumorDilLoss if num_classes > 1 else None,
            benign_dil_loss=BenignDilLoss if num_classes > 1 else None,
            boundary_weight=args.boundary_weight,
            tumor_dil_weight=args.tumor_dil_weight,
            benign_dil_weight=args.benign_dil_weight,
            num_classes=num_classes,
        )

        test_measure_mean, test_measure_std, val_detail = test(
            model,
            device,
            val_dataloader,
            epoch,
            perf_measure=perf,
            num_classes=num_classes,
        )

        if scheduler is not None:

            scheduler.step(test_measure_mean)

        state_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict()
            if args.mgpu == "false"
            else model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_stats["loss"],
            "val_score": test_measure_mean,
        }

        torch.save(
            state_dict,
            run_dir / "last_FCBFormer.pt",
        )

        if (
            prev_best_test is None
            or test_measure_mean > prev_best_test
        ):

            print("Saving best...")

            torch.save(
                state_dict,
                run_dir / "best_FCBFormer.pt",
            )

            prev_best_test = test_measure_mean


def get_args():

    parser = argparse.ArgumentParser(
        description="Train FCBFormer"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Kvasir", "CVC", "AIRA", "IDRiD"],
    )

    parser.add_argument(
        "--lesion",
        type=str,
        required=True,
        choices=["ma", "he", "ex", "se"],
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        dest="root",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        dest="lr",
    )

    parser.add_argument(
        "--learning-rate-scheduler",
        type=str,
        default="true",
        dest="lrs",
    )

    parser.add_argument(
        "--learning-rate-scheduler-minimum",
        type=float,
        default=1e-6,
        dest="lrs_min",
    )

    parser.add_argument(
        "--multi-gpu",
        type=str,
        default="false",
        dest="mgpu",
        choices=["true", "false"],
    )

    parser.add_argument("--boundary-weight", type=float, default=0.2)
    parser.add_argument("--tumor-dil-weight", type=float, default=0.2)
    parser.add_argument("--benign-dil-weight", type=float, default=0.1)
    parser.add_argument("--dil-iters", type=int, default=3)
    parser.add_argument("--dil-kernel", type=int, default=3)
    parser.add_argument("--prob-power", type=float, default=1.0)

    return parser.parse_args()


def main():

    args = get_args()

    run_dir = Path("outputs") / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed = 1337

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    train(args, run_dir)


if __name__ == "__main__":
    main()