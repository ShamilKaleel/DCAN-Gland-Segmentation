"""Training script for DCAN and FCN baseline.

Usage:
    uv run python scripts/train.py --config configs/dcan.yaml
    uv run python scripts/train.py --config configs/fcn_baseline.yaml
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dcan.data.dataset import GlaSDataset
from dcan.losses import AuxWeightScheduler, DCANLoss, FCNBaselineLoss
from dcan.models.dcan import DCAN
from dcan.models.fcn_baseline import FCNBaseline
from dcan.utils import load_config, save_config, seed_everything


def compute_class_weights(
    data_dir: Path, split: str, names: list[str], is_contour: bool = False, cap: float = 10.0
) -> torch.Tensor:
    """Capped inverse-frequency class weights on the training set.

    The background weight is normalized to 1.0 and the (rare) foreground/contour
    weight is the inverse-frequency ratio clipped to ``cap``. Capping is important
    for the contour class: an uncapped inverse-frequency weight (contours are a
    tiny fraction of pixels) drives the contour branch to predict contours almost
    everywhere, which over-suppresses foreground in the fusion step and over-splits
    glands -- the main reason the previous run's detection F1 collapsed.
    """
    import cv2
    from dcan.data.targets import instance_to_contour_mask, instance_to_object_mask

    counts = np.zeros(2, dtype=np.float64)
    for name in names:
        inst = cv2.imread(
            str(data_dir / split / "instances" / f"{name}.png"), cv2.IMREAD_UNCHANGED
        ).astype(np.uint16)
        if is_contour:
            mask = instance_to_contour_mask(inst)
        else:
            mask = instance_to_object_mask(inst)
        counts[0] += (mask == 0).sum()
        counts[1] += (mask == 1).sum()
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-8)
    weights = weights / weights.min()  # background -> 1.0
    weights = np.clip(weights, 1.0, cap)
    return torch.tensor(weights, dtype=torch.float32)


def load_split_names(path: str | Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def evaluate_val_dice(model, data_dir: Path, split: str, names: list[str], cfg: dict, device: str, is_dcan: bool) -> float:
    """Run the full inference + post-processing pipeline on the val fold and
    return the mean object-level Dice. Used for checkpoint selection, because
    validation cross-entropy overfits early and picks an underfit checkpoint.
    """
    import cv2
    from dcan.infer import overlap_tile_predict
    from dcan.metrics import object_dice
    from dcan.postprocess import fuse_predictions, postprocess_fcn, postprocess_mask

    model.eval()
    crop = cfg.get("crop_size", 480)
    stride = cfg.get("infer_stride", 240)
    t_o, t_c = cfg.get("t_o", 0.5), cfg.get("t_c", 0.5)
    sm, ma = cfg.get("smooth_radius", 3), cfg.get("min_area", 120)

    dices = []
    for name in names:
        img = cv2.cvtColor(cv2.imread(str(data_dir / split / "images" / f"{name}.png")), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(data_dir / split / "instances" / f"{name}.png"), cv2.IMREAD_UNCHANGED).astype(np.uint16)
        p_obj, p_cnt = overlap_tile_predict(model, img, crop_size=crop, stride=stride, device=device, is_dcan=is_dcan)
        if is_dcan:
            inst = postprocess_mask(fuse_predictions(p_obj, p_cnt, t_o, t_c), sm, ma)
        else:
            inst = postprocess_fcn(p_obj, t_o, sm, ma)
        dices.append(object_dice(gt, inst))
    return float(np.mean(dices)) if dices else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42))

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    is_dcan = cfg.get("model", "dcan") == "dcan"

    data_dir = Path(cfg["data_dir"])
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    # Load split
    train_names = load_split_names(cfg["train_split"])
    val_names = load_split_names(cfg["val_split"])

    # Class weights (capped inverse-frequency; see compute_class_weights)
    obj_cap = cfg.get("obj_weight_cap", 10.0)
    cnt_cap = cfg.get("cnt_weight_cap", 5.0)
    obj_w = compute_class_weights(
        data_dir, "train", train_names, is_contour=False, cap=obj_cap
    ).to(device)
    cnt_w = (
        compute_class_weights(
            data_dir, "train", train_names, is_contour=True, cap=cnt_cap
        ).to(device)
        if is_dcan
        else None
    )
    print(f"Class weights -> obj: {obj_w.tolist()}" + (f", cnt: {cnt_w.tolist()}" if cnt_w is not None else ""))

    # Datasets
    train_ds = GlaSDataset(
        data_dir,
        "train",
        names=train_names,
        crop_size=cfg.get("crop_size", 480),
        contour_radius=cfg.get("contour_radius", 3),
        is_train=True,
        crops_per_image=cfg.get("crops_per_image", 4),
    )
    val_ds = GlaSDataset(
        data_dir,
        "train",
        names=val_names,
        crop_size=cfg.get("crop_size", 480),
        contour_radius=cfg.get("contour_radius", 3),
        is_train=False,
        crops_per_image=1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 2),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    output_stride = cfg.get("output_stride", 8)
    if is_dcan:
        model = DCAN(pretrained=cfg.get("pretrained", True), output_stride=output_stride).to(device)
        criterion = DCANLoss(obj_weight=obj_w, cnt_weight=cnt_w)
    else:
        model = FCNBaseline(pretrained=cfg.get("pretrained", True), output_stride=output_stride).to(device)
        criterion = FCNBaselineLoss(obj_weight=obj_w)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 5e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=cfg.get("lr_patience", 10),
        min_lr=cfg.get("min_lr", 1e-7),
    )
    aux_scheduler = AuxWeightScheduler(
        init_w=cfg.get("w_a_init", 1.0),
        decay=cfg.get("w_a_decay", 0.1),
        step_iters=cfg.get("w_a_step", 10000),
        min_w=cfg.get("w_a_min", 1e-3),
    )

    use_amp = device == "cuda" and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(log_dir=str(run_dir / "logs"))

    # Append-mode logging: a re-launched run must not wipe an existing history
    # (the previous DCAN run was truncated this way). Existing logs are rotated.
    header = ["epoch", "iter", "train_loss", "val_loss", "val_dice", "lr", "w_a"]
    csv_path = run_dir / "train_log.csv"
    if csv_path.exists() and csv_path.stat().st_size > len(",".join(header) + "\n"):
        backup = run_dir / f"train_log.prev.{int(time.time())}.csv"
        csv_path.rename(backup)
        print(f"Existing train log preserved as {backup.name}")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)

    max_epochs = cfg.get("max_epochs", 100)
    accum_steps = max(1, cfg.get("accum_steps", 1))  # gradient accumulation (1 = off)
    freeze_bn = cfg.get("freeze_encoder_bn", False)
    eval_every = max(1, cfg.get("eval_every", 5))
    global_iter = 0
    best_val_dice = -1.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        if freeze_bn and hasattr(model, "freeze_encoder_bn"):
            model.freeze_encoder_bn()  # keep pretrained encoder BN in eval (bs=1 stability)
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")
        for step, batch in enumerate(pbar):
            # batch = (image, object target, contour target) from GlaSDataset.
            images = batch[0].to(device)
            obj_targets = batch[1].to(device)

            w_a = aux_scheduler(global_iter)  # current deep-supervision weight

            # Mixed precision (AMP) speeds up the forward/backward on the GPU.
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)  # -> obj/contour logits (+ aux maps)
                if is_dcan:
                    cnt_targets = batch[2].to(device)
                    losses = criterion(outputs, obj_targets, cnt_targets, w_a)
                else:
                    losses = criterion(outputs, obj_targets, w_a)

            # Gradient accumulation: divide so grads summed over `accum_steps`
            # micro-batches AVERAGE to one effective larger batch; step only every
            # accum_steps. (`scaler` is the AMP loss scaler.)
            scaler.scale(losses["total"] / accum_steps).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)   # unscale + SGD update
                scaler.update()          # adjust the AMP scale factor
                optimizer.zero_grad()

            loss_val = losses["total"].item()
            epoch_loss += loss_val
            n_batches += 1
            global_iter += 1

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                w_a=f"{w_a:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        # Flush any gradients left over when n_batches is not a multiple of accum_steps.
        if n_batches % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0].to(device)
                obj_targets = batch[1].to(device)
                outputs = model(images)
                if is_dcan:
                    cnt_targets = batch[2].to(device)
                    losses = criterion(outputs, obj_targets, cnt_targets, w_a=0.0)
                else:
                    losses = criterion(outputs, obj_targets, w_a=0.0)
                val_loss_sum += losses["total"].item()
                val_n += 1
        avg_val_loss = val_loss_sum / max(val_n, 1)

        scheduler.step(avg_val_loss)

        # Periodic object-Dice on the val fold for checkpoint selection.
        do_eval = (epoch % eval_every == 0) or (epoch == max_epochs)
        val_dice = evaluate_val_dice(model, data_dir, "train", val_names, cfg, device, is_dcan) if do_eval else None

        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("loss/val", avg_val_loss, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("w_a", w_a, epoch)
        if val_dice is not None:
            writer.add_scalar("val/dice", val_dice, epoch)

        csv_writer.writerow(
            [
                epoch,
                global_iter,
                f"{avg_train_loss:.6f}",
                f"{avg_val_loss:.6f}",
                f"{val_dice:.6f}" if val_dice is not None else "",
                f"{optimizer.param_groups[0]['lr']:.1e}",
                f"{w_a:.6f}",
            ]
        )
        csv_file.flush()

        print(
            f"  Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
            + (f", val_Dice={val_dice:.4f}" if val_dice is not None else "")
        )

        # Checkpoint selection by object-Dice, NOT validation cross-entropy:
        # val CE bottoms out (or overfits) early and would pick an under-segmenting
        # model, whereas object-Dice tracks the metric we actually report. `last.pt`
        # is always the latest epoch; `best.pt` is the highest val-Dice so far.
        ckpt = {
            "epoch": epoch,
            "global_iter": global_iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "val_dice": val_dice,
        }
        torch.save(ckpt, run_dir / "last.pt")
        if val_dice is not None and val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(ckpt, run_dir / "best.pt")
            print(f"  -> New best val Dice: {best_val_dice:.4f}")

    csv_file.close()
    writer.close()
    print(f"\nTraining complete. Checkpoints at {run_dir}")


if __name__ == "__main__":
    main()
