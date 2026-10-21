from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from .models import MLPAutoEncoder


@dataclass(frozen=True)
class EtaPosterior:
    """Gamma update and the precision used for training and allocation.

    Shape, rate, mean and variance always describe the unscaled Gamma.
    ``effective_mean`` is used by both neural and allocation updates.
    """

    shape: float
    rate: float
    mean: float
    variance: float
    eta_a: float
    eta_b: float
    lambda_reconstruction: float
    full_reconstruction_sse: float
    weighted_reconstruction_sse: float
    expected_cluster_sse: float
    expected_total_loss: float
    gibbs_precision: float
    eta_multiplier: float
    effective_mean: float


@torch.no_grad()
def encode_full_dataset(
    model: MLPAutoEncoder,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    n = len(loader.dataset)
    z_all = torch.empty((n, model.z_dim), dtype=torch.float32, device=device)
    for batch in loader:
        x = batch["image_pt"].to(device)
        indices = batch["index"].to(device=device, dtype=torch.long)
        z_all[indices] = model.encode(x)
    return z_all


def pooled_centered_variance(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError("z must be a nonempty two-dimensional tensor.")
    centered = z - z.mean(dim=0, keepdim=True)
    return centered.pow(2).mean()


@torch.no_grad()
def compute_cluster_stats(
    z: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = labels.to(device=z.device, dtype=torch.long)
    if labels.ndim != 1 or labels.shape[0] != z.shape[0]:
        raise ValueError("labels must be a length-n vector aligned with z.")
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive.")
    if torch.any(labels < 0) or torch.any(labels >= n_clusters):
        raise ValueError("labels must contain integers from 0 to K - 1.")

    counts = torch.bincount(
        labels.detach().cpu(), minlength=n_clusters
    ).to(device=z.device, dtype=z.dtype)
    if torch.any(counts <= 0):
        empty = torch.where(counts <= 0)[0].detach().cpu().tolist()
        raise RuntimeError(f"Empty clusters found: {empty}.")
    sums = torch.zeros((n_clusters, z.shape[1]), dtype=z.dtype, device=z.device)
    sums.index_add_(0, labels, z)
    means = sums / counts.unsqueeze(1)
    full_sse = (z - means[labels]).pow(2).sum()
    return means, counts, full_sse


@torch.no_grad()
def compute_partition_sses(
    z: torch.Tensor,
    partitions: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    if partitions.ndim != 2 or partitions.shape[1] != z.shape[0]:
        raise ValueError("partitions must have shape (M, n) aligned with z.")
    return torch.stack(
        [compute_cluster_stats(z, labels, n_clusters)[2] for labels in partitions]
    )


@torch.no_grad()
def initialize_labels_kmeans(
    z: torch.Tensor,
    n_clusters: int,
    seed: int,
) -> torch.Tensor:
    labels = KMeans(
        n_clusters=n_clusters,
        n_init=20,
        random_state=seed,
    ).fit_predict(z.detach().cpu().numpy())
    return torch.as_tensor(labels, dtype=torch.long, device=z.device)


def update_eta_posterior(
    *,
    n: int,
    p: int,
    d: int,
    eta_a: float,
    eta_b: float,
    lambda_reconstruction: float,
    full_reconstruction_sse: float,
    expected_cluster_sse: float,
    eta_shape_scale: float = 1.0,
    eta_multiplier: float = 1.0,
) -> EtaPosterior:
    """Gamma eta coordinate with shape ``eta_a + eta_shape_scale * n(p+d)/2``.

    ``eta_a`` and ``eta_b`` are the prior shape and rate. The positive
    sensitivity multiplier applies only to the extended-loss log-eta term.
    The rate is

        eta_b + lambda_R * R_SSE + E_qc[C_SSE].

    ``eta_multiplier`` does not change this Gamma distribution. It scales
    only the effective mean used by the algorithm. Non-unit values are an
    explicit sensitivity variant, not the original posterior coordinate update.
    """

    if n < 1 or p < 1 or d < 1:
        raise ValueError("n, p, and d must be positive integers.")
    scalars = {
        "eta_a": eta_a,
        "eta_b": eta_b,
        "eta_shape_scale": eta_shape_scale,
        "eta_multiplier": eta_multiplier,
        "lambda_reconstruction": lambda_reconstruction,
        "full_reconstruction_sse": full_reconstruction_sse,
        "expected_cluster_sse": expected_cluster_sse,
    }
    for name, value in scalars.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if eta_a <= 0 or eta_b <= 0 or lambda_reconstruction <= 0:
        raise ValueError("eta_a, eta_b and lambda_reconstruction must be positive.")
    if eta_shape_scale <= 0:
        raise ValueError("eta_shape_scale must be positive.")
    if eta_multiplier <= 0:
        raise ValueError("eta_multiplier must be positive.")
    if full_reconstruction_sse < 0 or expected_cluster_sse < 0:
        raise ValueError("SSE values must be non-negative.")

    shape = float(eta_a + eta_shape_scale * (n * (p + d) / 2.0))
    weighted_reconstruction = float(
        lambda_reconstruction * full_reconstruction_sse
    )
    expected_total = float(weighted_reconstruction + expected_cluster_sse)
    rate = float(eta_b + expected_total)
    mean = float(shape / rate)
    variance = float(shape / (rate * rate))
    effective_mean = float(eta_multiplier * mean)
    if not np.isfinite(effective_mean) or effective_mean <= 0:
        raise ValueError("Effective eta must be finite and positive.")
    return EtaPosterior(
        shape=shape,
        rate=rate,
        mean=mean,
        variance=variance,
        eta_a=float(eta_a),
        eta_b=float(eta_b),
        lambda_reconstruction=float(lambda_reconstruction),
        full_reconstruction_sse=float(full_reconstruction_sse),
        weighted_reconstruction_sse=weighted_reconstruction,
        expected_cluster_sse=float(expected_cluster_sse),
        expected_total_loss=expected_total,
        # Lambda_R never multiplies allocation precision: R cancels.
        gibbs_precision=effective_mean,
        eta_multiplier=float(eta_multiplier),
        effective_mean=effective_mean,
    )


def candidate_sse_increments(
    z_i: torch.Tensor,
    means: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Exact total-SSE increment from adding ``z_i`` to each cluster."""

    if means.ndim != 2 or z_i.ndim != 1 or means.shape[1] != z_i.shape[0]:
        raise ValueError("z_i and means have incompatible shapes.")
    if counts.ndim != 1 or counts.shape[0] != means.shape[0]:
        raise ValueError("counts must align with means.")
    if torch.any(counts <= 0):
        raise ValueError("Every candidate cluster must be nonempty.")
    counts_float = counts.to(device=means.device, dtype=means.dtype)
    dist2 = (means - z_i).pow(2).sum(dim=1)
    return counts_float / (counts_float + 1.0) * dist2


def allocation_probabilities(
    delta_cluster_sse: torch.Tensor,
    *,
    eta_mean: float,
) -> torch.Tensor:
    """Allocation probabilities after the common ``lambda_R R`` cancels."""

    if not np.isfinite(eta_mean) or eta_mean <= 0:
        raise ValueError("eta_mean must be finite and positive.")
    log_weights = -float(eta_mean) * delta_cluster_sse
    return torch.softmax(log_weights - torch.max(log_weights), dim=0)


@torch.no_grad()
def gibbs_sweep_exact_sse(
    z: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
    gibbs_precision: float,
    generator: torch.Generator | None = None,
    random_scan: bool = True,
) -> torch.Tensor:
    """One fixed-K sweep using the supplied effective precision times Delta C."""

    if gibbs_precision <= 0 or not np.isfinite(gibbs_precision):
        raise ValueError("gibbs_precision must be finite and positive.")
    labels = labels.clone().to(device=z.device, dtype=torch.long)
    means, counts_float, _ = compute_cluster_stats(z, labels, n_clusters)
    counts = counts_float.to(dtype=torch.long)
    order = (
        torch.randperm(z.shape[0], generator=generator, device="cpu").to(z.device)
        if random_scan
        else torch.arange(z.shape[0], device=z.device)
    )
    for i_tensor in order:
        i = int(i_tensor.item())
        old_k = int(labels[i].item())
        old_count = int(counts[old_k].item())
        if old_count <= 1:
            continue
        z_i = z[i]
        means[old_k] = (means[old_k] * old_count - z_i) / (old_count - 1)
        counts[old_k] -= 1
        delta_sse = candidate_sse_increments(z_i, means, counts)
        probabilities = allocation_probabilities(
            delta_sse, eta_mean=float(gibbs_precision)
        )
        new_k = int(
            torch.multinomial(
                probabilities.detach().cpu(), 1, generator=generator
            ).item()
        )
        new_count = int(counts[new_k].item())
        means[new_k] = (means[new_k] * new_count + z_i) / (new_count + 1)
        counts[new_k] += 1
        labels[i] = new_k
    return labels


@torch.no_grad()
def sample_fixed_eta_chains(
    *,
    z: torch.Tensor,
    partitions: torch.Tensor,
    n_clusters: int,
    eta_mean: float,
    burnin_per_chain: int,
    draws_per_chain: Sequence[int],
    thinning: int,
    generators: Sequence[torch.Generator],
    sweep_fn: Callable[..., torch.Tensor] = gibbs_sweep_exact_sse,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Advance independent fixed-K chains and retain draws in chain-major order."""

    if partitions.ndim != 2 or partitions.shape[1] != z.shape[0]:
        raise ValueError("partitions must have shape (M, n) aligned with z.")
    n_chains = partitions.shape[0]
    if len(draws_per_chain) != n_chains or len(generators) != n_chains:
        raise ValueError("draw counts and generators must align with chains.")
    if burnin_per_chain < 0 or thinning < 1:
        raise ValueError("burn-in must be non-negative and thinning positive.")
    if any(draw_count < 0 for draw_count in draws_per_chain):
        raise ValueError("draw counts must be non-negative.")

    return_device = partitions.device
    z_cpu = z.detach().cpu()
    states = partitions.detach().cpu().clone()
    stored: list[np.ndarray] = []
    chain_ids: list[int] = []
    for chain_index, draw_count in enumerate(draws_per_chain):
        for _ in range(burnin_per_chain):
            states[chain_index] = sweep_fn(
                z_cpu,
                states[chain_index],
                n_clusters,
                eta_mean,
                generators[chain_index],
                True,
            )
        for _ in range(draw_count):
            for _ in range(thinning):
                states[chain_index] = sweep_fn(
                    z_cpu,
                    states[chain_index],
                    n_clusters,
                    eta_mean,
                    generators[chain_index],
                    True,
                )
            stored.append(states[chain_index].numpy().copy())
            chain_ids.append(chain_index)
    if not stored:
        raise RuntimeError("Posterior sampling retained no draws.")
    return (
        np.stack(stored),
        np.asarray(chain_ids, dtype=np.int64),
        states.to(return_device),
    )


def co_clustering_matrix(samples: np.ndarray) -> np.ndarray:
    if samples.ndim != 2:
        raise ValueError("samples must have shape (S, n).")
    psm = np.zeros((samples.shape[1], samples.shape[1]), dtype=np.float64)
    for labels in samples:
        psm += (labels[:, None] == labels[None, :]).astype(np.float64)
    return psm / samples.shape[0]


def dahl_representative_partition(samples: np.ndarray, psm: np.ndarray) -> np.ndarray:
    best_index = 0
    best_loss = np.inf
    for index, labels in enumerate(samples):
        membership = (labels[:, None] == labels[None, :]).astype(np.float64)
        loss = float(np.square(membership - psm).sum())
        if loss < best_loss:
            best_loss = loss
            best_index = index
    return samples[best_index].copy()
