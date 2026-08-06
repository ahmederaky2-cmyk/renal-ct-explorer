from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)
from monai.utils import set_determinism

from renal_explorer.vessel_ml import sha256_file

KIPA22_TO_CANONICAL = [0, 5, 1, 4, 2]
CLASS_NAMES = ("background", "renal_vein", "kidney", "renal_artery", "tumor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate a compact 3D renal-vessel model on KiPA22 CTA."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--cache-rate", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--artery-dice-gate", type=float, default=0.70)
    parser.add_argument("--vein-dice-gate", type=float, default=0.65)
    return parser.parse_args()


def discover_cases(root: Path) -> list[dict[str, str]]:
    cases = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        image = case_dir / "imaging.nii.gz"
        label = case_dir / "segmentation.nii.gz"
        if image.is_file() and label.is_file():
            cases.append(
                {
                    "case_id": case_dir.name,
                    "image": str(image),
                    "label": str(label),
                }
            )
    if len(cases) < 10:
        raise ValueError(
            "At least ten authorized KiPA22 training cases are required. "
            "Do not use the KiTS23 dataset for this model."
        )
    return cases


def fold_split(
    cases: list[dict[str, str]],
    fold: int,
    folds: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if folds < 2 or not 0 <= fold < folds:
        raise ValueError("Fold configuration is invalid.")
    validation = [case for index, case in enumerate(cases) if index % folds == fold]
    training = [case for index, case in enumerate(cases) if index % folds != fold]
    return training, validation


def training_transforms(patches_per_case: int) -> Compose:
    keys = ("image", "label")
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(
                keys=keys,
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys="image",
                a_min=-100.0,
                a_max=1000.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=keys,
                source_key="label",
                select_fn=lambda values: values > 0,
                margin=24,
            ),
            SpatialPadd(keys=keys, spatial_size=(96, 96, 96)),
            RandCropByLabelClassesd(
                keys=keys,
                label_key="label",
                spatial_size=(96, 96, 96),
                ratios=(0.2, 4.0, 0.6, 5.0, 2.0),
                num_classes=5,
                num_samples=patches_per_case,
            ),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.15, prob=0.35),
            RandGaussianNoised(keys="image", std=0.015, prob=0.2),
            EnsureTyped(keys=keys),
        ]
    )


def validation_transforms() -> Compose:
    keys = ("image", "label")
    return Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(
                keys=keys,
                axcodes="RAS",
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys="image",
                a_min=-100.0,
                a_max=1000.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(
                keys=keys,
                source_key="label",
                select_fn=lambda values: values > 0,
                margin=24,
            ),
            SpatialPadd(keys=keys, spatial_size=(96, 96, 96)),
            EnsureTyped(keys=keys),
        ]
    )


def build_model() -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
        norm="INSTANCE",
    )


def _soft_erode(image: torch.Tensor) -> torch.Tensor:
    first = -functional.max_pool3d(-image, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    second = -functional.max_pool3d(-image, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    third = -functional.max_pool3d(-image, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.minimum(first, torch.minimum(second, third))


def _soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return functional.max_pool3d(image, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def _soft_skeleton(image: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    opened = _soft_dilate(_soft_erode(image))
    skeleton = functional.relu(image - opened)
    for _iteration in range(iterations):
        image = _soft_erode(image)
        opened = _soft_dilate(_soft_erode(image))
        delta = functional.relu(image - opened)
        skeleton = skeleton + functional.relu(delta - skeleton * delta)
    return skeleton


def vessel_cldice_loss(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    predicted_vessels = probabilities[:, 1:2] + probabilities[:, 3:4]
    truth_vessels = ((label == 1) | (label == 3)).float()
    predicted_skeleton = _soft_skeleton(predicted_vessels)
    truth_skeleton = _soft_skeleton(truth_vessels)
    smooth = 1e-5
    topology_precision = (
        torch.sum(predicted_skeleton * truth_vessels) + smooth
    ) / (torch.sum(predicted_skeleton) + smooth)
    topology_sensitivity = (
        torch.sum(truth_skeleton * predicted_vessels) + smooth
    ) / (torch.sum(truth_skeleton) + smooth)
    cldice = (2.0 * topology_precision * topology_sensitivity) / (
        topology_precision + topology_sensitivity + smooth
    )
    return 1.0 - cldice


def validation_dice(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    intersection = np.zeros(5, dtype=np.float64)
    denominator = np.zeros(5, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = sliding_window_inference(
                image,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
                mode="gaussian",
            )
            prediction = torch.argmax(logits, dim=1, keepdim=True)
            for class_index in range(1, 5):
                predicted_class = prediction == class_index
                true_class = label == class_index
                intersection[class_index] += float(
                    torch.count_nonzero(predicted_class & true_class).cpu()
                )
                denominator[class_index] += float(
                    torch.count_nonzero(predicted_class).cpu()
                    + torch.count_nonzero(true_class).cpu()
                )
    metrics = {}
    for class_index in range(1, 5):
        metrics[f"{CLASS_NAMES[class_index]}_dice"] = (
            2.0 * intersection[class_index] / denominator[class_index]
            if denominator[class_index] > 0
            else 1.0
        )
    metrics["mean_vessel_dice"] = float(
        np.mean((metrics["renal_artery_dice"], metrics["renal_vein_dice"]))
    )
    return metrics


def export_onnx(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    model.eval()
    example = torch.zeros((1, 1, 96, 96, 96), dtype=torch.float32, device=device)
    torch.onnx.export(
        model,
        example,
        str(path),
        input_names=["ct"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )


def write_manifest(
    output: Path,
    fold: int,
    training_cases: int,
    metrics: dict[str, float],
    artery_gate: float,
    vein_gate: float,
) -> None:
    validated = (
        metrics["renal_artery_dice"] >= artery_gate
        and metrics["renal_vein_dice"] >= vein_gate
    )
    manifest = {
        "schema_version": 1,
        "name": "KiPA22 compact 3D vessel U-Net",
        "version": "1.0.0",
        "dataset": "KiPA22 CTA",
        "dataset_url": "https://kipa22.grand-challenge.org/",
        "research_only": True,
        "validated": validated,
        "validation_fold": fold,
        "training_cases": training_cases,
        "target_spacing_mm": [1.0, 1.0, 1.0],
        "patch_size": [96, 96, 96],
        "overlap": 0.5,
        "intensity_clip_hu": [-100.0, 1000.0],
        "canonical_labels": KIPA22_TO_CANONICAL,
        "minimum_component_mm3": 10.0,
        "artifact_sha256": sha256_file(output / "model.onnx"),
        "validation_metrics": metrics,
        "validation_gates": {
            "renal_artery_dice": artery_gate,
            "renal_vein_dice": vein_gate,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if not validated:
        raise RuntimeError(
            "The model was exported but did not pass the artery/vein Dice gates. "
            "The Streamlit application will refuse to load it."
        )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_determinism(seed=args.seed)
    cases = discover_cases(args.data)
    train_cases, validation_cases = fold_split(cases, args.fold, args.folds)
    args.output.mkdir(parents=True, exist_ok=True)

    train_data = CacheDataset(
        train_cases,
        transform=training_transforms(args.patches_per_case),
        cache_rate=args.cache_rate,
        num_workers=args.num_workers,
    )
    validation_data = CacheDataset(
        validation_cases,
        transform=validation_transforms(),
        cache_rate=1.0,
        num_workers=args.num_workers,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    class_weights = torch.tensor((0.05, 3.0, 0.5, 4.0, 2.0), device=device)
    dice_loss = DiceLoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
    )
    cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_vessel_dice = -1.0
    checkpoint = args.output / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(image)
                segmentation_loss = 0.7 * dice_loss(logits, label)
                segmentation_loss = segmentation_loss + 0.3 * cross_entropy_loss(
                    logits,
                    label[:, 0].long(),
                )
                loss = 0.75 * segmentation_loss
                loss = loss + 0.25 * vessel_cldice_loss(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if epoch % args.validate_every == 0 or epoch == args.epochs:
            metrics = validation_dice(model, validation_loader, device)
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "loss": float(np.mean(losses)),
                        **metrics,
                    }
                ),
                flush=True,
            )
            if metrics["mean_vessel_dice"] > best_vessel_dice:
                best_vessel_dice = metrics["mean_vessel_dice"]
                torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    final_metrics = validation_dice(model, validation_loader, device)
    export_onnx(model, args.output / "model.onnx", device)
    write_manifest(
        args.output,
        args.fold,
        len(train_cases),
        final_metrics,
        args.artery_dice_gate,
        args.vein_dice_gate,
    )
    print(json.dumps({"status": "validated", **final_metrics}), flush=True)


if __name__ == "__main__":
    main()
