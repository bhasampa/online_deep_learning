# homework/models.py

from pathlib import Path
import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent

# Normalization stats for CNNPlanner
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


##############################################################################
# MODEL CLASSES
##############################################################################

class MLPPlanner(nn.Module):
    """
    A simple MLP that takes track_left + track_right boundary points
    and outputs future waypoints.
    """
    def __init__(self, n_track=10, n_waypoints=3, hidden_dim=128):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            hidden_dim (int): hidden size for the MLP
        """
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # We have track_left: (b, n_track, 2) and track_right: (b, n_track, 2).
        # That means total input_dim = 4*n_track (2 coords each for left + right).
        input_dim = 4 * n_track
        output_dim = n_waypoints * 2  # e.g. 3 * 2 = 6

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        track_left:  (b, n_track, 2)
        track_right: (b, n_track, 2)
        Returns: (b, n_waypoints, 2)
        """
        b = track_left.shape[0]
        # Flatten boundaries
        left_flat = track_left.view(b, -1)   # => (b, 2*n_track)
        right_flat = track_right.view(b, -1) # => (b, 2*n_track)
        x = torch.cat([left_flat, right_flat], dim=1)  # => (b, 4*n_track)

        out = self.net(x)                # => (b, n_waypoints*2)
        out = out.view(b, self.n_waypoints, 2)  # => (b, n_waypoints, 2)
        return out


class TransformerPlanner(nn.Module):
    """
    A small Transformer-based planner that attends over track_left + track_right boundary points
    to produce future waypoints.
    """
    def __init__(self, n_track=10, n_waypoints=3, d_model=64, nhead=4, num_layers=2):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
            d_model  (int): hidden size for the Transformer embeddings
            nhead    (int): number of heads in multi-head attention
            num_layers (int): number of TransformerDecoder layers
        """
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input embedding (2D coords -> d_model)
        self.input_embed = nn.Linear(2, d_model)
        # Query embeddings for each of the n_waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # A stack of TransformerDecoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final FC to map each decoded embedding to (x, y)
        self.output_fc = nn.Linear(d_model, 2)

    def forward(self, track_left: torch.Tensor, track_right: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        track_left:  (b, n_track, 2)
        track_right: (b, n_track, 2)
        Returns: (b, n_waypoints, 2)
        """
        b = track_left.shape[0]
        # Combine left/right => (b, 2*n_track, 2)
        boundaries = torch.cat([track_left, track_right], dim=1)

        # Embed => (b, 2*n_track, d_model)
        boundaries_emb = self.input_embed(boundaries)

        # Transformer expects (S, B, d_model), so swap batch <-> sequence dims
        boundaries_emb = boundaries_emb.transpose(0, 1)  # => (2*n_track, b, d_model)

        # Prepare queries => (n_waypoints, b, d_model)
        q_indices = torch.arange(self.n_waypoints, device=boundaries.device)
        queries = self.query_embed(q_indices)    # => (n_waypoints, d_model)
        queries = queries.unsqueeze(1).expand(-1, b, -1)  # => (n_waypoints, b, d_model)

        # Decode => (n_waypoints, b, d_model)
        decoded = self.decoder(tgt=queries, memory=boundaries_emb)

        # Map each of the n_waypoints embeddings to (x, y)
        out = self.output_fc(decoded)  # => (n_waypoints, b, 2)

        # Transpose back => (b, n_waypoints, 2)
        out = out.transpose(0, 1)
        return out


class CNNPlanner(nn.Module):
    """
    A CNN-based planner that takes an image (b, 3, H, W) and outputs (b, n_waypoints, 2).
    Default assumption: input shape ~ (b, 3, 96, 128) in [0,1] range
    """
    def __init__(self, n_waypoints=3):
        super().__init__()
        self.n_waypoints = n_waypoints

        # Store normalization stats in buffers
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Simple CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # => shape ~ (b, 128, H/8, W/8), e.g. (b, 128, 12, 16) for 96x128
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # => (b, 128, 1, 1)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2),  # => (b, n_waypoints*2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        image: (b, 3, H, W) in [0,1]
        return: (b, n_waypoints, 2)
        """
        # Normalize
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.features(x)  # => (b, 128, H/8, W/8)
        x = self.pool(x)      # => (b, 128, 1, 1)
        x = x.view(x.size(0), -1)  # => (b, 128)
        x = self.fc(x)        # => (b, n_waypoints*2)
        x = x.view(x.size(0), self.n_waypoints, 2)
        return x


##############################################################################
# MODEL FACTORY & UTILS
##############################################################################

class LinearPlanner(MLPPlanner):
    """
    Identical to MLPPlanner, just a different class name if you want to keep
    "linear_planner" as a separate entry in MODEL_FACTORY.
    """
    pass


MODEL_FACTORY = {
    # We’ll map "linear_planner" to the same MLP-based approach:
    "linear_planner": LinearPlanner,
    # If you prefer, you could just do: "linear_planner": MLPPlanner,
    
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
    Called by the grader (and your code) to load a pre-trained model by name.
    """
    if model_name not in MODEL_FACTORY:
        raise KeyError(
            f"Unknown model name '{model_name}'. "
            f"Valid keys: {list(MODEL_FACTORY.keys())}"
        )

    model = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model "
                "arguments match."
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(model)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return model


def save_model(model: torch.nn.Module) -> Path:
    """
    Use this function to save your model in train_planner.py
    """
    model_name = None
    for name, cls in MODEL_FACTORY.items():
        # In case we used a derived class (e.g. LinearPlanner inherits from MLPPlanner),
        # we can check via isinstance:
        if isinstance(model, cls):
            model_name = name
            break

    if model_name is None:
        raise ValueError(f"Model type '{type(model)}' not in MODEL_FACTORY.")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size in MB.
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
