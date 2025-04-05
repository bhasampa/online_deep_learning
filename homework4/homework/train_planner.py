import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from homework.datasets.road_dataset import RoadDataset
from homework.models import load_model, save_model
from homework.metrics import PlannerMetric

def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
    num_workers=4,
    lr=1e-3,
    batch_size=128,
    num_epoch=10,
):
    # 1. Create dataset/dataloaders
    train_set = RoadDataset(split="train", transform_pipeline=transform_pipeline)
    val_set = RoadDataset(split="val", transform_pipeline=transform_pipeline)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 2. Load/Create the model
    model = load_model(model_name, with_weights=False)  # or you can directly instantiate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. Create optimizer / loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(num_epoch):
        # ------------------
        # TRAIN
        # ------------------
        model.train()
        train_loss = 0.0

        # Create the PlannerMetric to track errors across the train epoch
        train_metric = PlannerMetric()

        for batch in train_loader:
            # Depending on transform_pipeline, the batch dict may differ
            # For "state_only" => "track_left", "track_right", "waypoints", "waypoints_mask"
            track_left = batch["track_left"].to(device)   # (B, 10, 2)
            track_right = batch["track_right"].to(device) # (B, 10, 2)
            waypoints_gt = batch["waypoints"].to(device)  # (B, 3, 2)
            mask = batch["waypoints_mask"].to(device)      # (B, 3)

            optimizer.zero_grad()

            # forward pass
            waypoints_pred = model(track_left=track_left, track_right=track_right)

            # compute loss (mask invalid waypoints)
            # mask => (B, 3), so expand to match shape (B, 3, 2)
            mask_3d = mask.unsqueeze(-1).expand(-1, -1, 2)
            loss = criterion(waypoints_pred[mask_3d], waypoints_gt[mask_3d])
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # update PlannerMetric
            train_metric.add(
                preds=waypoints_pred,
                labels=waypoints_gt,
                labels_mask=mask,
            )

        train_loss /= len(train_loader)
        train_stats = train_metric.compute()

        # ------------------
        # VALIDATION
        # ------------------
        model.eval()
        val_loss = 0.0
        val_metric = PlannerMetric()

        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints_gt = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                waypoints_pred = model(track_left=track_left, track_right=track_right)

                mask_3d = mask.unsqueeze(-1).expand(-1, -1, 2)
                loss = criterion(waypoints_pred[mask_3d], waypoints_gt[mask_3d])
                val_loss += loss.item()

                val_metric.add(
                    preds=waypoints_pred,
                    labels=waypoints_gt,
                    labels_mask=mask,
                )

        val_loss /= len(val_loader)
        val_stats = val_metric.compute()

        # Print some logs
        print(f"[Epoch {epoch+1}/{num_epoch}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
              f"train_lat={train_stats['lateral_error']:.4f} train_lon={train_stats['longitudinal_error']:.4f} | "
              f"val_lat={val_stats['lateral_error']:.4f} val_lon={val_stats['longitudinal_error']:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model)
            print("  -> New best model saved")


if __name__ == "__main__":
    train()  # run with default settings, or pass in your own
