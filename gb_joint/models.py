from __future__ import annotations

import torch
from torch import nn

from .architectures import SUPPORTED_ARCHITECTURES


class MLPAutoEncoder(nn.Module):
    """Autoencoder architectures for clustering.

    The ``dcec_conv`` option is the convolutional autoencoder from the DCEC
    MNIST implementation, adapted to accept and return flattened 784-vectors
    so the surrounding clustering code remains data-modality agnostic.
    ``mnist_conv`` is a separate, smaller two-layer CNN with sigmoid output.
    """

    def __init__(
        self,
        input_dim: int,
        z_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        architecture: str = "mlp",
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.architecture = str(architecture)

        if hidden_dims is not None:
            if self.architecture != "mlp":
                raise ValueError("hidden_dims is supported only for architecture='mlp'.")
            if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims or any(
                not isinstance(width, int) or isinstance(width, bool) or width < 1
                for width in hidden_dims
            ):
                raise ValueError("hidden_dims must be a nonempty sequence of positive integers.")

        if self.architecture == "linear":
            self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.z_dim))
            self.decoder = nn.Sequential(nn.Linear(self.z_dim, self.input_dim))
            return
        if self.architecture == "mnist_conv":
            if self.input_dim != 28 * 28:
                raise ValueError("architecture='mnist_conv' requires flattened 28x28 inputs.")
            if dropout != 0.0:
                raise ValueError("architecture='mnist_conv' requires dropout=0.")
            self.encoder = nn.Sequential(
                nn.Unflatten(1, (1, 28, 28)),
                nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, self.z_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.z_dim, 32 * 7 * 7),
                nn.ReLU(),
                nn.Unflatten(1, (32, 7, 7)),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
                nn.Flatten(),
            )
            return
        if self.architecture == "dcec_conv":
            if self.input_dim != 28 * 28:
                raise ValueError(
                    "architecture='dcec_conv' requires flattened 28x28 inputs."
                )
            if dropout != 0.0:
                raise ValueError("architecture='dcec_conv' requires dropout=0.")

            # DCEC encoder: Conv32(5,s2,same) -> Conv64(5,s2,same)
            # -> Conv128(3,s2,valid) -> FC(z).  The spatial sizes are
            # 28 -> 14 -> 7 -> 3, hence 128 * 3 * 3 inputs to the embedding.
            self.encoder = nn.Sequential(
                nn.Unflatten(1, (1, 28, 28)),
                nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * 3 * 3, self.z_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.z_dim, 128 * 3 * 3),
                nn.ReLU(),
                nn.Unflatten(1, (128, 3, 3)),
                nn.ConvTranspose2d(
                    128, 64, kernel_size=3, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    64,
                    32,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.ReLU(),
                nn.ConvTranspose2d(
                    32,
                    1,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                ),
                nn.Flatten(),
            )
            return
        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                "architecture must be 'linear', 'mlp', 'mlp_ln_gelu', "
                "'dcec_conv', or 'mnist_conv'."
            )

        if self.architecture == "mlp_ln_gelu":
            encoder_layers: list[nn.Module] = [
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
            ]
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            encoder_layers.append(nn.Linear(self.hidden_dim, self.z_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_layers: list[nn.Module] = [
                nn.Linear(self.z_dim, self.hidden_dim),
                nn.GELU(),
            ]
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            decoder_layers.append(nn.Linear(self.hidden_dim, self.input_dim))
            self.decoder = nn.Sequential(*decoder_layers)
            return

        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [self.hidden_dim]

        def relu_stack(dimensions: list[int]) -> nn.Sequential:
            layers: list[nn.Module] = []
            for index, (source, target) in enumerate(zip(dimensions, dimensions[1:])):
                layers.append(nn.Linear(source, target))
                if index < len(dimensions) - 2:
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
            # Both the latent layer and reconstruction output remain linear.
            return nn.Sequential(*layers)

        self.encoder = relu_stack([self.input_dim, *self.hidden_dims, self.z_dim])
        self.decoder = relu_stack([self.z_dim, *reversed(self.hidden_dims), self.input_dim])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z.reshape(z.shape[0], -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))
