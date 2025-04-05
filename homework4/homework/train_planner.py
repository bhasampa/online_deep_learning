# homework/train_planner.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from typing import Optional

# Suppose these are local modules you have
from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import compute_waypoint_error  # Or however your metrics are structured

def train(
    model_name: str,
    transform_pipeline: str,
    num_workers: int,
    lr: float,
    batch_size: int,
    num_epoch: int,
    save_dir: str = "./checkpoints",
):
    """
    Trains a model (MLP, Transformer, CNN, etc.) depending on model_name.
    """

    # -------------------------------------------------------------------------
    # 1. Create dataset and dataloaders
    # -------------------------------------------------------------------------
    # transform_pipeline could be something like "state_only" (no images) or "image_only",
    # depending on how your RoadDataset is set up.

    train_set = RoadDataset(
        split="train",
        transform_pipeline=transform_pipeline,
    )
    val_set = RoadDataset(
        split="val",
        transform_pipeline=transform_pipeline,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)

    # -------------------------------------------------------------------------
    # 2. Initialize the model
    # -------------------------------------------------------------------------
    if model_name == "linear_planner" or model_name == "mlp_planner":
        model = MLPPlanner()
    elif model_name == "transformer_planner":
        model = TransformerPlanner()
    elif model_name == "cnn_planner":
        model = CNNPlanner()
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # -------------------------------------------------------------------------
    # 3. Initialize optimizer and loss function
    # -------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Typically for a regression to (x, y) waypoints, an L1 or L2 loss is used.
    # L2 is common (MSE). We'll use MSE as an example:
    criterion = nn.MSELoss()

    # -------------------------------------------------------------------------
    # 4. Training loop
    # -------------------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(num_epoch):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            # The RoadDataset might return different things depending on the transform pipeline
            # For "state_only":
            #   left_boundaries: (B, 10, 2)
            #   right_boundaries: (B, 10, 2)
            #   waypoints: (B, 3, 2)
            #   mask: (B, 3)
            # For "image_only":
            #   image: (B, 3, H, W)
            #   ...
            # etc.

            # Example for "state_only":
            left_boundaries = batch["track_left"].to(device)   # (B, 10, 2)
            right_boundaries = batch["track_right"].to(device) # (B, 10, 2)
            waypoints_gt = batch["waypoints"].to(device)        # (B, 3, 2)
            waypoints_mask = batch["waypoints_mask"].to(device) # (B, 3)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            # MLPPlanner / TransformerPlanner expect (left, right) as input.
            # CNNPlanner expects images. So you might do:
            if model_name in ["linear_planner", "mlp_planner", "transformer_planner"]:
                waypoints_pred = model(left_boundaries, right_boundaries)
            elif model_name == "cnn_planner":
                images = batch["image"].to(device)  # (B, 3, 96, 128)
                waypoints_pred = model(images)
            else:
                raise ValueError(f"Unknown model_name={model_name}")

            # Compute loss
            # If the dataset has a mask for invalid waypoints, you could apply it:
            # mask shape: (B, 3), so expand to (B, 3, 2) if you want to mask out x,y
            mask_3d = waypoints_mask.unsqueeze(-1).expand_as(waypoints_pred)  # (B, 3, 2)
            valid_waypoints_pred = waypoints_pred[mask_3d]
            valid_waypoints_gt = waypoints_gt[mask_3d]

            loss = criterion(valid_waypoints_pred, valid_waypoints_gt)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ---------------------------------------------------------------------
        # 5. Validation loop
        # ---------------------------------------------------------------------
        model.eval()
        total_val_loss = 0.0

        # Example metrics from your `compute_waypoint_error` or similar
        total_longitudinal_error = 0.0
        total_lateral_error = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if model_name in ["linear_planner", "mlp_planner", "transformer_planner"]:
                    left_boundaries = batch["track_left"].to(device)
                    right_boundaries = batch["track_right"].to(device)
                    waypoints_gt = batch["waypoints"].to(device)
                    waypoints_mask = batch["waypoints_mask"].to(device)
                    waypoints_pred = model(left_boundaries, right_boundaries)
                else:
                    images = batch["image"].to(device)
                    waypoints_gt = batch["waypoints"].to(device)
                    waypoints_mask = batch["waypoints_mask"].to(device)
                    waypoints_pred = model(images)

                mask_3d = waypoints_mask.unsqueeze(-1).expand_as(waypoints_pred)
                valid_waypoints_pred = waypoints_pred[mask_3d]
                valid_waypoints_gt = waypoints_gt[mask_3d]

                loss = criterion(valid_waypoints_pred, valid_waypoints_gt)
                total_val_loss += loss.item()

                # Evaluate your lateral & longitudinal error, e.g.:
                #   For each sample in the batch, compute error and accumulate
                #   (B, 3, 2) => compute the difference in x (lateral) and y (longitudinal)
                #   You might use a function from `metrics.py` like:
                #   lat_err, lon_err = compute_waypoint_error(waypoints_pred, waypoints_gt, waypoints_mask)
                #   total_longitudinal_error += sum(lon_err)
                #   total_lateral_error += sum(lat_err)
                #   n_samples += ???

        avg_val_loss = total_val_loss / len(val_loader)
        # If you summed errors over all samples, you can divide by n_samples here
        # avg_lon_err = total_longitudinal_error / n_samples
        # avg_lat_err = total_lateral_error / n_samples

        print(
            f"[Epoch {epoch+1}/{num_epoch}] "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}"
            # f" lon_err={avg_lon_err:.3f}, lat_err={avg_lat_err:.3f}"
        )

        # ---------------------------------------------------------------------
        # 6. Save the best model
        # ---------------------------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"{model_name}_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best val loss. Saved model to {ckpt_path}")

