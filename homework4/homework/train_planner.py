# homework/train_planner.py

import torch
import torch.optim as optim
import torch.nn as nn

from homework.datasets.road_dataset import load_data
from homework.models import load_model, save_model
from homework.metrics import PlannerMetric


def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
    data_root="drive_data",
    lr=1e-3,
    batch_size=128,
    num_epoch=10,
    num_workers=4,
):
    """
    Train a model (MLP, Transformer, or CNN) given a transform pipeline and data location.

    Args:
        model_name         (str): "linear_planner", "mlp_planner", "transformer_planner", or "cnn_planner"
        transform_pipeline (str): "default", "state_only", or another pipeline from road_dataset.py
        data_root          (str): path to data folder, e.g. "drive_data" which has "train"/"val" subfolders
        lr               (float): learning rate
        batch_size        (int): batch size
        num_epoch         (int): number of training epochs
        num_workers       (int): number of data-loading workers
    """

    # ----------------------------------------------------
    # 1. Create DataLoaders (train and val)
    # ----------------------------------------------------
    train_loader = load_data(
        dataset_path=f"{data_root}/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for training
    )

    val_loader = load_data(
        dataset_path=f"{data_root}/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
    )

    print(f"Train loader: {len(train_loader.dataset)} samples")
    print(f"Val   loader: {len(val_loader.dataset)} samples")

    # ----------------------------------------------------
    # 2. Create Model, Optimizer, Loss
    # ----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, with_weights=False)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    # ----------------------------------------------------
    # 3. Training Loop
    # ----------------------------------------------------
    for epoch in range(num_epoch):
        # -----------------------
        # TRAIN
        # -----------------------
        model.train()
        train_loss = 0.0
        train_metric = PlannerMetric()  # tracks L1, lat/lon errors

        for batch in train_loader:
            optimizer.zero_grad()

            # If using MLP or Transformer or Linear Planner, load track boundaries
            if model_name in ["mlp_planner", "transformer_planner", "linear_planner"]:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints_gt = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                waypoints_pred = model(track_left=track_left, track_right=track_right)

            else:  # "cnn_planner"
                images = batch["image"].to(device)
                waypoints_gt = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                waypoints_pred = model(image=images)

            mask_3d = mask.unsqueeze(-1).expand_as(waypoints_gt)  # (B, n_waypoints, 2)
            loss = criterion(waypoints_pred[mask_3d], waypoints_gt[mask_3d])
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_metric.add(
                preds=waypoints_pred, 
                labels=waypoints_gt, 
                labels_mask=mask
            )

        train_loss /= len(train_loader)
        train_stats = train_metric.compute()  # returns dict of lat_err, lon_err, etc.

        # -----------------------
        # VALIDATION
        # -----------------------
        model.eval()
        val_loss = 0.0
        val_metric = PlannerMetric()

        with torch.no_grad():
            for batch in val_loader:
                if model_name in ["mlp_planner", "transformer_planner", "linear_planner"]:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    waypoints_gt = batch["waypoints"].to(device)
                    mask = batch["waypoints_mask"].to(device)

                    waypoints_pred = model(track_left=track_left, track_right=track_right)

                else:  # "cnn_planner"
                    images = batch["image"].to(device)
                    waypoints_gt = batch["waypoints"].to(device)
                    mask = batch["waypoints_mask"].to(device)

                    waypoints_pred = model(image=images)

                mask_3d = mask.unsqueeze(-1).expand_as(waypoints_gt)
                loss = criterion(waypoints_pred[mask_3d], waypoints_gt[mask_3d])
                val_loss += loss.item()

                val_metric.add(
                    preds=waypoints_pred, 
                    labels=waypoints_gt, 
                    labels_mask=mask
                )

        val_loss /= len(val_loader)
        val_stats = val_metric.compute()

        print(
            f"[Epoch {epoch+1}/{num_epoch}] "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} | "
            f"Train Lat={train_stats['lateral_error']:.4f}, "
            f"Train Lon={train_stats['longitudinal_error']:.4f} | "
            f"Val Lat={val_stats['lateral_error']:.4f}, "
            f"Val Lon={val_stats['longitudinal_error']:.4f}"
        )

        # Save if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model)
            print("  -> New best model saved.")
