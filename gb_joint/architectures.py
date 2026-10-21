from __future__ import annotations


SUPPORTED_ARCHITECTURES = (
    "linear",
    "mlp",
    "mlp_ln_gelu",
    "dcec_conv",
    "mnist_conv",
)

ARCHITECTURE_TOPOLOGIES = {
    "linear": "p -> d -> p",
    "mlp": (
        "p -> hidden_dims (ReLU after each hidden layer) -> d; "
        "mirrored decoder with linear latent and reconstruction output"
    ),
    "mlp_ln_gelu": (
        "p -> hidden -> LayerNorm -> GELU -> d; d -> hidden -> GELU -> p"
    ),
    "dcec_conv": (
        "flattened 28x28 -> DCEC convolutional encoder -> d -> "
        "DCEC transposed-convolutional decoder -> flattened 28x28"
    ),
    "mnist_conv": (
        "1x28x28 -> Conv16(4,2,1) -> Conv32(4,2,1) -> 1568 -> d; "
        "d -> 1568 -> 32x7x7 -> ConvTranspose16(4,2,1) -> "
        "ConvTranspose1(4,2,1) -> sigmoid -> flattened 28x28"
    ),
}
