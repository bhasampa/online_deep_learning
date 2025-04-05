# homework/models.py
from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): hidden dimension for the MLP
        """
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # We have track_left (b, n_track, 2) and track_right (b, n_track, 2)
        # so total input_dim = 2 * n_track * 2 = 4 * n_track
        input_dim = 4 * n_track    # e.g. 4*10=40
        output_dim = n_waypoints * 2  # e.g. 3*2=6

        # A simple 2-layer MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        Args:
            track_left  (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: (b, n_waypoints, 2)
        """
        b = track_left.shape[0]

        # Flatten each boundary into (b, n_track*2)
        left_flat = track_left.view(b, -1)   # (b, n_track*2)
        right_flat = track_right.view(b, -1) # (b, n_track*2)

        # Concatenate along dim=1 => shape (b, 4*n_track)
        x = torch.cat([left_flat, right_flat], dim=1)

        # Run through MLP => (b, n_waypoints*2)
        out = self.net(x)

        # Reshape => (b, n_waypoints, 2)
        out = out.view(b, self.n_waypoints, 2)
        return out


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        """
        Args:
            n_track     (int): number of points in each side of the track
            n_waypoints (int): how many waypoints to predict
            d_model     (int): hidden dimension for the Transformer
            nhead       (int): number of heads in multi-head attention
            num_layers  (int): number of TransformerDecoder layers
        """
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Embedding for each boundary point (2D -> d_model)
        self.input_embed = nn.Linear(2, d_model)

        # Combined left+right => 2*n_track points total
        # We will feed these as "memory" to the Transformer

        # Query embeddings: we want n_waypoints queries
        # that will attend over the boundary embeddings
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # A stack of TransformerDecoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear layer to convert each decoded embedding => (x, y)
        self.output_fc = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict waypoints from the left and right boundaries.

        Args:
            track_left  (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: (b, n_waypoints, 2)
        """
        b = track_left.shape[0]

        # 1) Combine left and right => shape (b, 2*n_track, 2)
        boundaries = torch.cat([track_left, track_right], dim=1)  # (b, 2n_track, 2)

        # 2) Embed => (b, 2n_track, d_model)
        boundaries_emb = self.input_embed(boundaries)

        # 3) The Transformer expects (S, B, d_model), so transpose
        # S = 2n_track, B = batch size
        boundaries_emb = boundaries_emb.transpose(0, 1)  # => (2n_track, b, d_model)

        # 4) Prepare queries: n_waypoints queries, each is an embedding (d_model)
        #    shape => (n_waypoints, d_model), expanded to (n_waypoints, b, d_model)
        q_indices = torch.arange(self.n_waypoints, device=boundaries.device)
        query_embeddings = self.query_embed(q_indices)            # (n_waypoints, d_model)
        query_embeddings = query_embeddings.unsqueeze(1).expand(-1, b, -1)
        # => (n_waypoints, b, d_model)

        # 5) Decode => shape (n_waypoints, b, d_model)
        decoded = self.decoder(
            tgt=query_embeddings,
            memory=boundaries_emb,
        )

        # 6) Output => (n_waypoints, b, 2)
        out = self.output_fc(decoded)

        # 7) Transpose to (b, n_waypoints, 2)
        out = out.transpose(0, 1)  # => (b, n_waypoints, 2)
        return out


class CNNPlanner(nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()
        self.n_waypoints = n_waypoints

        # For normalization
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Simple CNN layers
        # Input: (b, 3, H=96, W=128)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Now shape ~ (b, 128, H/8, W/8) => (b, 128, 12, 16) for 96x128
        )

        # Global average pool => (b, 128, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final FC => (b, n_waypoints * 2)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) in [0, 1]

        Returns:
            torch.FloatTensor: (b, n_waypoints, 2)
        """
        # Normalize
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through CNN
        x = self.features(x)              # (b, 128, 12, 16)
        x = self.pool(x)                  # (b, 128, 1, 1)
        x = x.view(x.size(0), -1)         # (b, 128)

        # FC => (b, n_waypoints * 2)
        x = self.fc(x)

        # Reshape => (b, n_waypoints, 2)
        x = x.view(x.size(0), self.n_waypoints, 2)
        return x


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n
    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
