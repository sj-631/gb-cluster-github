from __future__ import annotations

import csv
import platform
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sklearn
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
from torch.utils.data import DataLoader

from .architectures import ARCHITECTURE_TOPOLOGIES
from .clustering import (
    EtaPosterior,
    co_clustering_matrix,
    compute_cluster_stats,
    compute_partition_sses,
    dahl_representative_partition,
    encode_full_dataset,
    gibbs_sweep_exact_sse,
    initialize_labels_kmeans,
    pooled_centered_variance,
    sample_fixed_eta_chains,
    update_eta_posterior,
)
from .config import TrainConfig
from .data import NpyClusteringDataset
from .losses import (
    calibrate_lambda_reconstruction,
    full_data_total_estimate,
    total_loss_neural_objective,
)
from .models import MLPAutoEncoder
from .posterior import (
    PosteriorSummaryResult,
    empirical_expected_vi,
    ensure_salso_available,
    parse_salso_metadata,
    relabel_consecutive,
    run_salso_vi,
    vi_medoid_representative_partition,
)
from .scale import project_latent_scale
from .utils import choose_device, data_manifest, save_json, set_seed, sha256_file, source_manifest


class TotalLossTrainer:
    """Fit the autoencoder and partition distribution by Monte Carlo updates.

    The implementation uses ``L = lambda_R * R_SSE + C_SSE`` throughout.
    Each neural minibatch uses full-data centroids at the current network
    parameters. Their derivative cancels in the full profiled SSE, so detaching
    current means gives an unbiased minibatch gradient without stale centres.
    Network weights are point estimates (the manuscript's small-epsilon
    plug-in approximation), not samples from a finite-variance weight posterior.
    """

    @staticmethod
    def validate_runtime_dependencies(config: TrainConfig) -> dict:
        if config.posterior_summary == "salso":
            script = Path(__file__).resolve().parents[1] / "scripts" / "salso_vi.R"
            return ensure_salso_available(script)
        return {"method": config.posterior_summary, "salso_used": False}

    def __init__(self, config: TrainConfig, dataset: NpyClusteringDataset) -> None:
        config.validate()
        self.validate_dataset_contract(config, dataset)
        self.posterior_environment = self.validate_runtime_dependencies(config)
        self.config = config
        self.dataset = dataset
        self.device = choose_device(config.device)
        if config.latent_scale_projection and self.device.type != "cpu":
            raise ValueError("latent_scale_projection is validated on CPU only; set device='cpu'.")
        set_seed(config.seed)
        self.started_utc = datetime.now(timezone.utc).isoformat()

        self.output_dir = Path(config.output_dir).expanduser().resolve()
        if any((self.output_dir / name).exists() for name in (
            "config.json", "run_summary.json", "model_state.pt",
        )):
            raise FileExistsError(f"Refusing to overwrite an existing run: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(self.output_dir / "config.json", config.to_dict())
        project_root = Path(__file__).resolve().parents[1]
        self.source_manifest = source_manifest(project_root)
        save_json(self.output_dir / "source_manifest.json", self.source_manifest)
        save_json(self.output_dir / "data_manifest.json",
                  data_manifest(config.x_path, config.y_path))

        self.environment = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "torch_version": torch.__version__,
            "device_requested": config.device,
            "training_device_resolved": str(self.device),
            "posterior_sampling_device": "cpu",
            "posterior_summary_environment": self.posterior_environment,
            "started_utc": self.started_utc,
        }
        self._save_diagnostic_json("environment.json", self.environment)
        self._save_diagnostic_json(
            "model_definition.json",
            {
                "loss": "lambda_reconstruction * raw_R_SSE + raw_C_SSE",
                "lambda_calibration_stage": config.lambda_calibration_stage,
                "eta_initialization": config.eta_initialization,
                "lambda_calibration_target": (
                    "median(lambda_R * ||grad_encoder R|| / "
                    "||grad_encoder C||) = r0"
                ),
                "warmup": "reconstruction-only raw total SSE; no prior penalty",
                "eta_parameterization": "shape-rate",
                "eta_shape": "eta_a + eta_shape_scale * n * (p + d) / 2",
                "eta_a": config.eta_a,
                "eta_shape_scale": config.eta_shape_scale,
                "eta_multiplier": config.eta_multiplier,
                "eta_rate": "eta_b + lambda_R * R_SSE + E_qc[C_SSE]",
                "effective_eta": "eta_multiplier * E_q[eta]",
                "eta_multiplier_interpretation": (
                    "base Gamma update unchanged; a non-unit multiplier is an "
                    "algorithmic sensitivity variant, not exact coordinate ascent "
                    "for the original generalized posterior"
                ),
                "neural_coordinate": "effective_eta * (lambda_R * R + E_qc[C]) + prior",
                "allocation_coordinate": "exp(-effective_eta * Delta C)",
                "partition_learning_rate": "none",
                "latent_scale_projection": self._scale_definition(),
                "architecture": config.architecture,
                "architecture_topology": ARCHITECTURE_TOPOLOGIES[
                    config.architecture
                ],
                "centroid_refresh": "full data, current network, before every neural minibatch",
                "network_approximation": "point-estimate plug-in; no finite-epsilon weight sampling",
                "optimizer": config.optimizer,
                "stopping_rule": "fixed outer_iterations budget, not a convergence guarantee",
                "initialization": config.partition_initialization,
                "initial_partition_path": config.to_dict()["initial_partition_path"],
                "n_chains": config.n_chains,
                "gibbs_sweeps_B": config.gibbs_sweeps_per_outer,
                "retained_mc_partitions_M": config.mc_partitions,
                "methodology_variants": self._methodology_variants(),
            },
        )

        self.train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=config.num_workers,
        )
        self.eval_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
        )
        self.model = MLPAutoEncoder(
            input_dim=int(dataset.x.shape[1]),
            z_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            architecture=config.architecture,
        ).to(self.device)
        optimizer_class = torch.optim.SGD if config.optimizer == "sgd" else torch.optim.Adam
        self.pretrain_optimizer = optimizer_class(
            self.model.parameters(), lr=config.lr_pretrain
        )
        self.joint_optimizer = optimizer_class(
            self.model.parameters(), lr=config.lr_joint
        )
        self.lambda_reconstruction = float("nan")
        self.history: List[Dict[str, float]] = []
        self.partition_history: List[np.ndarray] = []
        self.pretrain_metrics: Dict[str, float] = {}
        self.centroid_refresh_count = 0
        self.pretrain_gradient_steps = 0
        self.joint_gradient_steps = 0
        self.scale_history: list[dict[str, object]] = []
        self.initial_partition_metrics: Dict[str, float] = {}

        self.chain_generators: List[torch.Generator] = []
        self.chain_seeds: List[int] = []
        for chain_index in range(config.n_chains):
            generator = torch.Generator(device="cpu")
            # Use a separate, reproducible seed for each chain.
            chain_seed = config.seed + 104729 * (chain_index + 1)
            generator.manual_seed(chain_seed)
            self.chain_seeds.append(chain_seed)
            self.chain_generators.append(generator)
        # Loading a supplied partition consumes no model or chain randomness.
        # External reference labels (dataset.y) are never used to initialize.
        self.provided_initial_labels: torch.Tensor | None = None
        self.initialization_manifest: dict[str, object] = {
            "method": config.partition_initialization,
            "reference_labels_used_for_initialization": False,
            "n_chains": config.n_chains,
            "latent_kmeans_diagnostic_is_actual_initialization": (
                config.partition_initialization == "latent_kmeans"
            ),
        }
        if config.partition_initialization == "provided":
            self._load_provided_initial_partition()
        save_json(self.output_dir / "initial_partition_manifest.json", self.initialization_manifest)

    def _load_provided_initial_partition(self) -> None:
        path = Path(self.config.initial_partition_path).expanduser().resolve()
        labels = np.load(path, allow_pickle=False)
        if not isinstance(labels, np.ndarray):
            if hasattr(labels, "close"):
                labels.close()
            raise ValueError("The provided initial partition must be one .npy array, not an archive.")
        if labels.shape != (len(self.dataset),):
            raise ValueError("The provided initial partition must have shape (n,).")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("The provided initial partition must have an integer dtype.")
        if np.any(labels < 0):
            raise ValueError("The provided initial partition cannot contain negative labels.")
        values, normalized = np.unique(labels, return_inverse=True)
        if len(values) != self.config.n_clusters:
            raise ValueError("The provided initial partition must contain exactly K nonempty clusters.")
        normalized = normalized.astype(np.int64, copy=False)
        self.provided_initial_labels = torch.from_numpy(normalized.copy())
        self.initialization_manifest.update({
            "path": str(path),
            "sha256": sha256_file(path),
            "input_dtype": str(labels.dtype),
            "original_label_values": values.tolist(),
            "normalized_label_values": list(range(self.config.n_clusters)),
            "relabeling": "sorted distinct input labels mapped to consecutive zero-based labels",
            "relabeling_changed_values": not np.array_equal(labels, normalized),
            "cluster_sizes": np.bincount(normalized).tolist(),
            "selection": "supplied by caller; no reference-label-based selection in trainer",
        })

    def _provided_partitions(self) -> torch.Tensor:
        if self.provided_initial_labels is None:
            raise RuntimeError("No provided initial partition was loaded.")
        return self.provided_initial_labels.to(self.device).unsqueeze(0).repeat(
            self.config.n_chains, 1
        )

    def _scale_definition(self) -> dict[str, object] | str:
        if not self.config.latent_scale_projection:
            return "none"
        return {
            "target_pooled_centered_variance": 1.0,
            "timing": "after warmup, before eta/after-warmup calibration; after every joint neural epoch, before eta",
            "transformation": "encoder final weights and bias times s; decoder first weights divided by s",
            "centering_or_whitening": False,
            "optimizer_state": "unchanged",
            "device_support": "cpu only",
            "interpretation": "added scale constraint; not objective/prior invariant",
        }

    def _project_scale(self, stage: str) -> tuple[dict[str, object], torch.Tensor]:
        rng_before = torch.get_rng_state().clone()
        record, z = project_latent_scale(self.model, self.eval_loader, self.device)
        rng_unchanged = torch.equal(rng_before, torch.get_rng_state())
        if not rng_unchanged:
            raise RuntimeError("Latent scale projection unexpectedly changed the Torch RNG state.")
        record.update(stage=stage, rng_unchanged=rng_unchanged)
        self.scale_history.append(record)
        self._save_diagnostic_json("latent_scale_history.json", self.scale_history)
        return record, z

    def _methodology_variants(self) -> list[str]:
        """Record settings that differ from the Algorithm 1 reference values."""
        variants = []
        for key, reference in (
            ("lambda_calibration_stage", "after_warmup"),
            ("eta_initialization", "pretrained_loss"),
            ("partition_sampler", "single_chain_sequential"),
            ("posterior_summary", "salso"),
            ("optimizer", "sgd"),
            ("eta_shape_scale", 1.0),
            ("eta_multiplier", 1.0),
            ("initial_chain_burnin_sweeps", 0),
            ("partition_initialization", "latent_kmeans"),
        ):
            if getattr(self.config, key) != reference:
                variants.append(f"{key}={getattr(self.config, key)} (reference: {reference})")
        if self.config.latent_scale_projection:
            variants.append("latent pooled centered variance projected to 1 after warmup and each joint epoch")
        return variants

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[seed={self.config.seed}] {message}", flush=True)

    def _save_diagnostic_json(self, filename: str, values: Dict[str, object]) -> None:
        if self.config.save_diagnostics:
            save_json(self.output_dir / filename, values)

    def _save_diagnostic_array(self, filename: str, values: np.ndarray) -> None:
        if self.config.save_diagnostics:
            np.save(self.output_dir / filename, values)

    def _save_training_history(self) -> None:
        rows = self.history
        if not self.config.save_diagnostics:
            columns = (
                "outer_iteration", "exact_reconstruction_sse",
                "exact_weighted_reconstruction_sse", "mean_full_cluster_sse",
                "exact_expected_total_loss", "eta_mean_used", "eta_mean_new",
                "eta_multiplier", "eta_effective_used", "eta_effective_new",
                "iteration_seconds",
            )
            rows = [{key: row[key] for key in columns} for row in rows]
            for compact, full in zip(rows, self.history):
                compact.update({
                    key: value for key, value in full.items()
                    if key.startswith("partition_ari_")
                })
        self._write_rows(self.output_dir / "training_history.csv", rows)

    @staticmethod
    def validate_dataset_contract(
        config: TrainConfig, dataset: NpyClusteringDataset
    ) -> None:
        if config.n_clusters > len(dataset):
            raise ValueError("n_clusters cannot exceed the number of observations.")
        if config.architecture in {"dcec_conv", "mnist_conv"}:
            if int(dataset.x.shape[1]) != 28 * 28:
                raise ValueError(
                    f"{config.architecture} requires flattened 28x28 inputs (784 columns)."
                )
            minimum = float(dataset.x.min().item())
            maximum = float(dataset.x.max().item())
            if minimum < 0 or maximum > 1:
                raise ValueError(f"{config.architecture} requires input pixels in [0, 1].")

    def _metrics(self, labels: torch.Tensor | np.ndarray) -> Dict[str, float]:
        if self.dataset.y is None:
            return {}
        pred = np.asarray(labels.detach().cpu() if torch.is_tensor(labels) else labels)
        truth = self.dataset.y.cpu().numpy()
        return {
            "ari": float(adjusted_rand_score(truth, pred)),
            "nmi": float(normalized_mutual_info_score(truth, pred)),
            "ri": float(rand_score(truth, pred)),
        }

    def _partition_ari_metrics(self, partitions: torch.Tensor) -> Dict[str, float]:
        """Evaluate retained Gibbs states against the reference labels.

        A joint iteration retains M states before its neural update. These
        labels do not change during that update. The last-state ARI and the
        mean across M states describe the training partitions, rather than
        the final SALSO representative. This diagnostic does not affect fitting
        or consume training random numbers.
        """
        if self.dataset.y is None:
            return {}
        states = partitions.detach().cpu().numpy()
        if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] != len(self.dataset):
            raise ValueError("partitions must be a nonempty (M, n) matrix.")
        truth = self.dataset.y.cpu().numpy()
        values = np.asarray([adjusted_rand_score(truth, labels) for labels in states])
        return {
            "partition_ari_last": float(values[-1]),
            "partition_ari_mean": float(values.mean()),
            "partition_ari_min": float(values.min()),
            "partition_ari_max": float(values.max()),
        }

    def _raw_network_prior(self) -> torch.Tensor:
        """Unnormalised Gaussian penalty, outside ``E(eta) * L``."""

        encoder_sum = torch.zeros((), device=self.device)
        decoder_sum = torch.zeros((), device=self.device)
        for parameter in self.model.encoder.parameters():
            encoder_sum = encoder_sum + parameter.pow(2).sum()
        for parameter in self.model.decoder.parameters():
            decoder_sum = decoder_sum + parameter.pow(2).sum()
        return 0.5 * (
            self.config.encoder_prior_precision * encoder_sum
            + self.config.decoder_prior_precision * decoder_sum
        )

    def _initial_partitions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        z = encode_full_dataset(self.model, self.eval_loader, self.device)
        partitions = torch.stack(
            [
                initialize_labels_kmeans(
                    z,
                    n_clusters=self.config.n_clusters,
                    seed=self.config.seed + 1009 * chain_index,
                )
                for chain_index in range(self.config.n_chains)
            ],
            dim=0,
        )
        return partitions, z

    def _partition_means(
        self,
        z: torch.Tensor,
        partitions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack(
            [
                compute_cluster_stats(z, labels, self.config.n_clusters)[0]
                for labels in partitions
            ],
            dim=0,
        )

    def _mc_batch_cluster_sse(
        self,
        z: torch.Tensor,
        indices: torch.Tensor,
        partitions: torch.Tensor,
        means: torch.Tensor,
    ) -> torch.Tensor:
        partitions_device = partitions.to(self.device)
        targets = torch.stack(
            [
                means[m][partitions_device[m, indices]]
                for m in range(partitions.shape[0])
            ],
            dim=0,
        )
        # Raw sum over observations and latent coordinates, MC mean over q(c).
        return (z.unsqueeze(0) - targets).pow(2).flatten(start_dim=1).sum(dim=1).mean()

    @staticmethod
    def _gradient_norm(gradients: Iterable[torch.Tensor]) -> float:
        value = torch.sqrt(sum(g.detach().pow(2).sum() for g in gradients))
        return float(value.item())

    def _calibrate_lambda(
        self, partitions: torch.Tensor | None = None
    ) -> dict[str, object]:
        """Calibrate lambda_R without changing training RNG or parameters."""

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = self.model.training
        try:
            if self.config.save_diagnostics and self.config.lambda_calibration_stage == "before_warmup":
                torch.save(self.model.state_dict(), self.output_dir / "model_state_calibration.pt")
            if partitions is None and self.config.partition_initialization == "provided":
                partitions = self._provided_partitions()
            if partitions is None:
                partitions, z_full = self._initial_partitions()
                partition_source = "deterministic latent K-means states (pre-warmup variant)"
            else:
                partitions = partitions.detach().clone().to(self.device)
                z_full = encode_full_dataset(self.model, self.eval_loader, self.device)
                partition_source = "actual pretrained latent K-means initial partition"
            if self.config.partition_initialization == "provided":
                partition_source = (
                    "provided GB initial partition, evaluated in random initial latent space"
                    if self.config.lambda_calibration_stage == "before_warmup" else
                    "provided GB initial partition, evaluated in pretrained latent space"
                )
            means = self._partition_means(z_full, partitions).detach()
            encoder_parameters = tuple(self.model.encoder.parameters())
            ratios: List[float] = []
            reconstruction_norms: List[float] = []
            clustering_norms: List[float] = []
            cosines: List[float] = []
            n = len(self.dataset)

            self.model.eval()
            for batch_index, batch in enumerate(self.eval_loader):
                if (
                    self.config.calibration_batches is not None
                    and batch_index >= self.config.calibration_batches
                ):
                    break
                x = batch["image_pt"].to(self.device)
                indices = batch["index"].to(self.device, dtype=torch.long)
                z = self.model.encode(x)
                x_hat = self.model.decoder(z)
                scale = n / float(x.shape[0])
                reconstruction_total = scale * (x_hat - x).pow(2).sum()
                clustering_total = scale * self._mc_batch_cluster_sse(
                    z, indices, partitions, means
                )
                gradients_r = torch.autograd.grad(
                    reconstruction_total,
                    encoder_parameters,
                    retain_graph=True,
                    allow_unused=False,
                )
                gradients_c = torch.autograd.grad(
                    clustering_total,
                    encoder_parameters,
                    retain_graph=False,
                    allow_unused=False,
                )
                norm_r = self._gradient_norm(gradients_r)
                norm_c = self._gradient_norm(gradients_c)
                if not np.isfinite(norm_r) or not np.isfinite(norm_c):
                    raise RuntimeError("Calibration produced a non-finite gradient norm.")
                if norm_r <= 0 or norm_c <= 0:
                    raise RuntimeError("A zero gradient norm prevents calibration.")
                ratios.append(norm_r / (norm_c + self.config.gradient_epsilon))
                reconstruction_norms.append(norm_r)
                clustering_norms.append(norm_c)
                dot = sum(
                    (gradient_r.detach() * gradient_c.detach()).sum()
                    for gradient_r, gradient_c in zip(gradients_r, gradients_c)
                )
                cosines.append(float(dot.item()) / (norm_r * norm_c))
            if not ratios:
                raise RuntimeError("The calibration loader produced no batches.")
            self.lambda_reconstruction = calibrate_lambda_reconstruction(
                ratios,
                target_gradient_ratio=self.config.target_gradient_ratio,
            )
            initial_labels = partitions.detach().cpu().numpy()
            self._save_diagnostic_array(
                ("persistent_partitions_random_initialization.npy"
                 if self.config.lambda_calibration_stage == "before_warmup"
                 else "calibration_partitions_pretrained.npy"),
                initial_labels,
            )
            result: dict[str, object] = {
                "weight_placement": "reconstruction",
                "calibration_stage": (
                    "random network initialization before warmup"
                    if self.config.lambda_calibration_stage == "before_warmup"
                    else "pretrained network after reconstruction warmup"
                ),
                "partition_initialization": partition_source,
                "partition_states": int(partitions.shape[0]),
                "calibration_batches": len(ratios),
                "raw_total_sse": True,
                "full_data_batch_scale": "n / B for both R and C",
                "partitions_and_centroids_fixed_during_pass": True,
                "network_fixed_during_pass": True,
                "gradient_epsilon": self.config.gradient_epsilon,
                "target_gradient_ratio_r0": self.config.target_gradient_ratio,
                "gradient_ratio_definition": "||grad R|| / ||grad C||",
                "gradient_ratio_median": float(np.median(ratios)),
                "gradient_ratio_mean": float(np.mean(ratios)),
                "gradient_ratio_min": float(np.min(ratios)),
                "gradient_ratio_max": float(np.max(ratios)),
                "lambda_reconstruction": self.lambda_reconstruction,
                "achieved_weighted_ratio": float(
                    self.lambda_reconstruction * np.median(ratios)
                ),
                "reconstruction_gradient_norm_median": float(
                    np.median(reconstruction_norms)
                ),
                "clustering_gradient_norm_median": float(np.median(clustering_norms)),
                "gradient_cosine_median": float(np.median(cosines)),
                "raw_gradient_ratios": ratios,
                "rng_state_restored_after_calibration": True,
            }
            if self.config.partition_initialization == "provided":
                result["initial_labels_sha256"] = self.initialization_manifest["sha256"]
                result["no_precalibration_scale_projection"] = not self.scale_history
            self._save_diagnostic_json("lambda_reconstruction_calibration.json", result)
            return result
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            self.model.train(was_training)

    def _train_reconstruction_epoch(self) -> float:
        self.model.train()
        totals = 0.0
        observations = 0
        n = len(self.dataset)
        for batch in self.train_loader:
            x = batch["image_pt"].to(self.device)
            reconstruction_total = full_data_total_estimate(
                (self.model(x) - x).pow(2).sum(), n=n, batch_size=x.shape[0]
            )
            objective = reconstruction_total
            if not torch.isfinite(objective):
                raise FloatingPointError(
                    "Non-finite warm-up loss; reduce the learning rate. "
                    "Raw-SSE SGD and Adam learning rates are not interchangeable."
                )
            self.pretrain_optimizer.zero_grad(set_to_none=True)
            objective.backward()
            self.pretrain_optimizer.step()
            self.pretrain_gradient_steps += 1
            totals += float(objective.item()) * x.shape[0]
            observations += int(x.shape[0])
        return totals / max(observations, 1)

    def pretrain(self) -> torch.Tensor:
        for epoch in range(self.config.pretrain_epochs):
            epoch_start = time.perf_counter()
            loss = self._train_reconstruction_epoch()
            self._log(
                f"[Pretrain {epoch + 1}/{self.config.pretrain_epochs}] "
                f"objective={loss:.6g} (R only) | "
                f"time={time.perf_counter() - epoch_start:.1f}s"
            )
        if self.config.save_diagnostics:
            checkpoint_name = (
                "model_state_pretrained_before_scale.pt"
                if self.config.latent_scale_projection else "model_state_pretrained.pt"
            )
            torch.save(self.model.state_dict(), self.output_dir / checkpoint_name)
        partitions, z = self._initial_partitions()
        if self.config.partition_sampler == "single_chain_sequential":
            # One latent K-means start, also used for calibration and eta.
            partitions = partitions[:1]
        self._save_diagnostic_array(
            "latent_pretrained_before_scale.npy"
            if self.config.latent_scale_projection else "latent_pretrained.npy",
            z.cpu().numpy(),
        )
        self._save_diagnostic_array(
            "persistent_partitions_pretrained.npy",
            partitions.cpu().numpy(),
        )
        self._save_diagnostic_array(
            "labels_kmeans_pretrained.npy",
            partitions[0].cpu().numpy(),
        )
        self.pretrain_metrics = self._metrics(partitions[0])
        self._save_diagnostic_json(
            "metrics_kmeans_pretrained.json", self.pretrain_metrics,
        )
        self._log("[Pretrain complete] " + " | ".join(
            f"{name.upper()}={value:.4f}" for name, value in self.pretrain_metrics.items()
        ))
        # Compute the K-means diagnostic for both initialization modes to keep
        # forward-pass order and RNG use consistent.
        if self.config.partition_initialization == "provided":
            partitions = self._provided_partitions()
        self.initial_partition_metrics = self._metrics(partitions[0])
        self.initialization_manifest.update({
            "metrics": self.initial_partition_metrics,
            "metrics_role": "post-initialization evaluation only; not used to select labels",
            "stage": "after warmup, before optional scale projection and initial eta",
        })
        save_json(self.output_dir / "initial_partition_manifest.json", self.initialization_manifest)
        self._save_diagnostic_array("persistent_partitions_gibbs_initial.npy", partitions.cpu().numpy())
        if self.config.latent_scale_projection:
            projection, z = self._project_scale("after_warmup")
            if self.config.save_diagnostics:
                torch.save(self.model.state_dict(), self.output_dir / "model_state_pretrained.pt")
            self._save_diagnostic_array("latent_pretrained.npy", z.cpu().numpy())
            self._save_diagnostic_json("latent_scale_pretrained.json", projection)
            self._log(
                f"[Scale after warmup] variance={projection['variance_before']:.8g}"
                f" -> {projection['variance_after']:.8g}; factor={projection['scale_factor']:.8g}"
            )
        return partitions

    @torch.no_grad()
    def _full_reconstruction_sse(self) -> float:
        self.model.eval()
        total = 0.0
        for batch in self.eval_loader:
            x = batch["image_pt"].to(self.device)
            total += float((self.model(x) - x).pow(2).sum().item())
        return total

    @torch.no_grad()
    def _evaluate_total_loss(
        self,
        partitions: torch.Tensor,
    ) -> Tuple[EtaPosterior, torch.Tensor, torch.Tensor, float]:
        z = encode_full_dataset(self.model, self.eval_loader, self.device)
        sses = compute_partition_sses(z, partitions.to(self.device), self.config.n_clusters)
        reconstruction_sse = self._full_reconstruction_sse()
        eta = update_eta_posterior(
            n=len(self.dataset),
            p=int(self.dataset.x.shape[1]),
            d=self.config.z_dim,
            eta_a=self.config.eta_a,
            eta_b=self.config.eta_b,
            eta_shape_scale=self.config.eta_shape_scale,
            eta_multiplier=self.config.eta_multiplier,
            lambda_reconstruction=self.lambda_reconstruction,
            full_reconstruction_sse=reconstruction_sse,
            expected_cluster_sse=float(sses.mean().item()),
        )
        return eta, z, sses, reconstruction_sse

    def _initialize_eta(self, partitions: torch.Tensor) -> EtaPosterior:
        """Choose only the starting q(eta), without advancing training RNGs."""

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = self.model.training
        try:
            if self.config.eta_initialization == "pretrained_loss":
                eta, _, _, _ = self._evaluate_total_loss(partitions)
                source = "actual pretrained reconstruction and initial-partition losses"
            else:
                eta = update_eta_posterior(
                    n=len(self.dataset),
                    p=int(self.dataset.x.shape[1]),
                    d=self.config.z_dim,
                    eta_a=self.config.eta_a,
                    eta_b=self.config.eta_b,
                    eta_shape_scale=self.config.eta_shape_scale,
                    eta_multiplier=self.config.eta_multiplier,
                    lambda_reconstruction=self.lambda_reconstruction,
                    full_reconstruction_sse=0.0,
                    expected_cluster_sse=0.0,
                )
                source = "zero-loss extended-Gamma initialization (non-manuscript variant)"
            self._save_diagnostic_json(
                "eta_initialization.json", {"source": source, **asdict(eta)}
            )
            return eta
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            self.model.train(was_training)

    def _advance_partitions(
        self,
        z: torch.Tensor,
        partitions: torch.Tensor,
        *,
        eta_mean: float,
        sweeps: int,
    ) -> torch.Tensor:
        """Advance allocations at an already scaled effective eta; R cancels."""

        return_device = partitions.device
        z_cpu = z.detach().cpu()
        states = partitions.detach().cpu().clone()
        if self.config.partition_sampler == "single_chain_sequential":
            if sweeps < self.config.mc_partitions or sweeps % self.config.mc_partitions:
                raise ValueError("Need an equal positive number of sweeps per retained MC draw.")
            # Continue the last draw from the preceding outer iteration.
            # Only one RNG stream and one evolving chain state are used.
            state = states[-1].clone()
            draws = []
            for _ in range(self.config.mc_partitions):
                for _ in range(sweeps // self.config.mc_partitions):
                    state = gibbs_sweep_exact_sse(
                        z=z_cpu,
                        labels=state,
                        n_clusters=self.config.n_clusters,
                        gibbs_precision=float(eta_mean),
                        generator=self.chain_generators[0],
                        random_scan=True,
                    )
                draws.append(state.clone())
            return torch.stack(draws).to(return_device)
        for chain_index in range(states.shape[0]):
            for _ in range(sweeps):
                states[chain_index] = gibbs_sweep_exact_sse(
                    z=z_cpu,
                    labels=states[chain_index],
                    n_clusters=self.config.n_clusters,
                    gibbs_precision=float(eta_mean),
                    generator=self.chain_generators[chain_index],
                    random_scan=True,
                )
        return states.to(return_device)

    def _train_joint_epoch(
        self,
        partitions: torch.Tensor,
        eta_mean: float,
    ) -> Dict[str, float]:
        """Use the supplied effective eta without applying a second multiplier."""
        totals = np.zeros(7, dtype=np.float64)
        observations = 0
        n = len(self.dataset)
        for batch in self.train_loader:
            # Recompute at the parameters used by this gradient step.
            # Supported architectures are deterministic (dropout=0, no BN);
            # eval and train therefore define the same representation.
            z_full = encode_full_dataset(self.model, self.eval_loader, self.device)
            means = self._partition_means(z_full, partitions.to(self.device)).detach()
            self.centroid_refresh_count += 1
            self.model.train()
            x = batch["image_pt"].to(self.device)
            indices = batch["index"].to(self.device, dtype=torch.long)
            z = self.model.encode(x)
            x_hat = self.model.decoder(z)
            reconstruction_total = full_data_total_estimate(
                (x_hat - x).pow(2).sum(), n=n, batch_size=x.shape[0]
            )
            clustering_total = full_data_total_estimate(
                self._mc_batch_cluster_sse(z, indices, partitions, means),
                n=n,
                batch_size=x.shape[0],
            )
            prior = self._raw_network_prior()
            objective = total_loss_neural_objective(
                eta_mean=eta_mean,
                lambda_reconstruction=self.lambda_reconstruction,
                reconstruction_total=reconstruction_total,
                clustering_total=clustering_total,
                network_prior=prior,
            )
            if not torch.isfinite(objective):
                raise FloatingPointError(
                    "Non-finite joint loss; reduce the learning rate. "
                    "No clipping or rescaling was silently applied."
                )
            self.joint_optimizer.zero_grad(set_to_none=True)
            objective.backward()
            self.joint_optimizer.step()
            self.joint_gradient_steps += 1

            weighted_reconstruction = self.lambda_reconstruction * reconstruction_total
            joint_total = weighted_reconstruction + clustering_total
            batch_size = int(x.shape[0])
            totals += batch_size * np.asarray(
                [
                    objective.item(),
                    reconstruction_total.item(),
                    weighted_reconstruction.item(),
                    clustering_total.item(),
                    joint_total.item(),
                    (float(eta_mean) * joint_total).item(),
                    prior.item(),
                ]
            )
            observations += batch_size
        average = totals / max(observations, 1)
        training = {
            "training_objective": float(average[0]),
            "training_reconstruction_total": float(average[1]),
            "training_weighted_reconstruction_total": float(average[2]),
            "training_clustering_total": float(average[3]),
            "training_joint_total": float(average[4]),
            "training_eta_times_joint": float(average[5]),
            "raw_network_prior": float(average[6]),
        }
        if self.config.latent_scale_projection:
            projection, _ = self._project_scale(f"joint_epoch_{len(self.scale_history)}")
            training.update({
                "scale_variance_before": projection["variance_before"],
                "scale_variance_after": projection["variance_after"],
                "scale_factor": projection["scale_factor"],
                "scale_max_abs_reconstruction_change": projection["max_abs_reconstruction_change"],
            })
        return training

    def _draw_from_chains(
        self,
        *,
        z: torch.Tensor,
        partitions: torch.Tensor,
        eta_mean: float,
        burnin_per_chain: int,
        draws_per_chain: Sequence[int],
        thinning: int,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        samples, _, states = sample_fixed_eta_chains(
            z=z,
            partitions=partitions,
            n_clusters=self.config.n_clusters,
            eta_mean=eta_mean,
            burnin_per_chain=burnin_per_chain,
            draws_per_chain=draws_per_chain,
            thinning=thinning,
            generators=self.chain_generators,
        )
        return samples, states

    @staticmethod
    def _write_rows(path: Path, rows: List[Dict[str, float]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _posterior_summary(
        self,
        samples: np.ndarray,
        psm: np.ndarray | None,
    ) -> Tuple[np.ndarray, PosteriorSummaryResult, str]:
        if self.config.posterior_summary == "salso":
            filename = "representative_partition_salso.npy"
            representative, stdout = run_salso_vi(
                samples_path=self.output_dir / "partition_samples.npy",
                output_path=self.output_dir / filename,
                n_clusters=self.config.n_clusters,
                n_runs=self.config.salso_n_runs,
                n_cores=self.config.salso_n_cores,
                seed=self.config.seed,
                script_path=Path(__file__).resolve().parents[1] / "scripts" / "salso_vi.R",
            )
            summary = PosteriorSummaryResult(
                method="SALSO randomized greedy search under VI loss",
                realized_clusters=int(np.unique(representative).size),
                empirical_expected_vi=empirical_expected_vi(representative, samples),
                candidate_index=None,
                candidate_space="partitions of n observations (1 through n clusters); not restricted to retained draws",
                salso_n_runs=self.config.salso_n_runs,
                salso_stdout=stdout,
                salso_metadata=parse_salso_metadata(stdout),
            )
        elif self.config.posterior_summary == "vi_medoid":
            representative, index, expected_vi = vi_medoid_representative_partition(samples)
            filename = "representative_partition_vi_medoid.npy"
            summary = PosteriorSummaryResult(
                method="fixed-K posterior-sample VI medoid",
                realized_clusters=int(np.unique(representative).size),
                empirical_expected_vi=expected_vi,
                candidate_index=index,
                candidate_space="retained fixed-K allocation draws",
            )
        else:
            if psm is None:
                raise RuntimeError("Dahl's summary requires a computed PSM.")
            representative = relabel_consecutive(
                dahl_representative_partition(samples, psm)
            )
            filename = "representative_partition_dahl.npy"
            summary = PosteriorSummaryResult(
                method="Dahl posterior-similarity criterion",
                realized_clusters=int(np.unique(representative).size),
                empirical_expected_vi=empirical_expected_vi(representative, samples),
                candidate_index=None,
                candidate_space="retained fixed-K allocation draws",
            )
        self._save_diagnostic_array(filename, representative)
        np.save(self.output_dir / "representative_partition.npy", representative)
        save_json(self.output_dir / "posterior_summary.json", summary.to_dict())
        return representative, summary, (
            filename if self.config.save_diagnostics else "representative_partition.npy"
        )

    def fit(self) -> Dict[str, object]:
        start = time.time()
        self._log(
            f"[Start] n={len(self.dataset)} | p={self.dataset.x.shape[1]} | "
            f"K={self.config.n_clusters} | d={self.config.z_dim} | "
            f"architecture={self.config.architecture} | device={self.device} | "
            f"optimizer={self.config.optimizer} | "
            f"lr_pretrain={self.config.lr_pretrain:.6g} | lr_joint={self.config.lr_joint:.6g}"
        )
        calibration = None
        if self.config.lambda_calibration_stage == "before_warmup":
            self._log("[Calibration] Computing reconstruction weight at random initialization...")
            calibration = self._calibrate_lambda()
            self._log(f"[Calibration complete] lambda_R={self.lambda_reconstruction:.6g}")
        partitions = self.pretrain().to(self.device)
        if self.config.lambda_calibration_stage == "after_warmup":
            self._log("[Calibration] Computing reconstruction weight after warmup...")
            calibration = self._calibrate_lambda(partitions)
            self._log(f"[Calibration complete] lambda_R={self.lambda_reconstruction:.6g}")
        assert calibration is not None
        eta = self._initialize_eta(partitions)
        self._log(
            f"[Eta initialization] {self.config.eta_initialization}: "
            f"base_mean={eta.mean:.6g} | multiplier={eta.eta_multiplier:.6g} | "
            f"effective_eta={eta.effective_mean:.6g}"
        )
        z = encode_full_dataset(self.model, self.eval_loader, self.device)
        if self.config.initial_chain_burnin_sweeps:
            partitions = self._advance_partitions(
                z,
                partitions,
                eta_mean=eta.effective_mean,
                sweeps=self.config.initial_chain_burnin_sweeps,
            )

        for outer in range(self.config.outer_iterations):
            iteration_start = time.perf_counter()
            eta_used = float(eta.mean)
            effective_eta_used = float(eta.effective_mean)
            z = encode_full_dataset(self.model, self.eval_loader, self.device)
            partitions = self._advance_partitions(
                z,
                partitions,
                eta_mean=effective_eta_used,
                sweeps=self.config.gibbs_sweeps_per_outer,
            )
            partition_metrics = self._partition_ari_metrics(partitions)
            training: Dict[str, float] = {}
            for epoch in range(self.config.joint_epochs_per_outer):
                training = self._train_joint_epoch(partitions, effective_eta_used)
                if self.config.joint_epochs_per_outer > 1:
                    self._log(
                        f"[Joint {outer + 1}/{self.config.outer_iterations}, "
                        f"epoch {epoch + 1}/{self.config.joint_epochs_per_outer}] "
                        f"objective={training['training_objective']:.6g} | "
                        f"base_eta={eta_used:.6g} | effective_eta={effective_eta_used:.6g}"
                    )
            eta, z, sses, reconstruction_sse = self._evaluate_total_loss(partitions)
            record = {
                "outer_iteration": float(outer + 1),
                "eta_mean_used": eta_used,
                "eta_multiplier": eta.eta_multiplier,
                "eta_effective_used": effective_eta_used,
                **partition_metrics,
                **training,
                "lambda_reconstruction": self.lambda_reconstruction,
                "exact_reconstruction_sse": reconstruction_sse,
                "exact_weighted_reconstruction_sse": eta.weighted_reconstruction_sse,
                "mean_full_cluster_sse": float(sses.mean().item()),
                "sd_full_cluster_sse": float(sses.std(unbiased=False).item()),
                "exact_expected_total_loss": eta.expected_total_loss,
                "eta_shape_new": eta.shape,
                "eta_rate_new": eta.rate,
                "eta_mean_new": eta.mean,
                "eta_effective_new": eta.effective_mean,
                "eta_variance_new": eta.variance,
                "gibbs_precision_new": eta.gibbs_precision,
                "latent_centered_variance": float(pooled_centered_variance(z).item()),
                "centroid_refresh_count_cumulative": float(self.centroid_refresh_count),
                "iteration_seconds": time.perf_counter() - iteration_start,
            }
            self.history.append(record)
            self._save_training_history()
            if self.config.save_diagnostics:
                self.partition_history.append(partitions.cpu().numpy().copy())
            ari_text = (
                f"ARI_last={partition_metrics['partition_ari_last']:.6f} | "
                f"ARI_mean{partitions.shape[0]}={partition_metrics['partition_ari_mean']:.6f} | "
                if partition_metrics else ""
            )
            self._log(
                f"[Joint {outer + 1}/{self.config.outer_iterations}] "
                f"{ari_text}"
                f"R={reconstruction_sse:.6g} | lambda_R*R={eta.weighted_reconstruction_sse:.6g} | "
                f"C={float(sses.mean().item()):.6g} | loss={eta.expected_total_loss:.6g} | "
                f"base_eta={eta_used:.6g} -> {eta.mean:.6g} | "
                f"effective_eta={effective_eta_used:.6g} -> {eta.effective_mean:.6g} | "
                f"time={record['iteration_seconds']:.1f}s"
            )

        final_z = encode_full_dataset(self.model, self.eval_loader, self.device)
        final_reconstruction_sse = self._full_reconstruction_sse()
        # Freeze the last joint-training eta for both burn-in and retained draws.
        sampling_eta = eta
        self._log(
            f"[Posterior] Network fixed; base_eta={sampling_eta.mean:.6g} | "
            f"effective_eta={sampling_eta.effective_mean:.6g}"
        )

        base_draws, remainder = divmod(
            self.config.posterior_samples, self.config.n_chains
        )
        draws_per_chain = [
            base_draws + (1 if index < remainder else 0)
            for index in range(self.config.n_chains)
        ]
        self._log(
            f"[Sampling] {self.config.n_chains} chains | "
            f"burn-in={self.config.posterior_burnin_sweeps} sweeps/chain | "
            f"retained={self.config.posterior_samples} | thinning={self.config.posterior_thinning}"
        )
        if self.config.partition_sampler == "single_chain_sequential":
            partitions = partitions[-1:].clone()
        samples, partitions = self._draw_from_chains(
            z=final_z,
            partitions=partitions,
            eta_mean=sampling_eta.effective_mean,
            burnin_per_chain=self.config.posterior_burnin_sweeps,
            draws_per_chain=draws_per_chain,
            thinning=self.config.posterior_thinning,
        )
        np.save(self.output_dir / "partition_samples.npy", samples)
        compute_psm = (
            self.config.psm_mode == "full"
            or self.config.posterior_summary == "dahl"
            or (
                self.config.psm_mode == "auto"
                and len(self.dataset) <= self.config.psm_max_n
            )
        )
        psm = co_clustering_matrix(samples) if compute_psm else None
        if psm is not None:
            np.save(self.output_dir / "co_clustering_matrix.npy", psm)
        self._save_diagnostic_json(
            "psm_manifest.json",
            {
                "requested_mode": self.config.psm_mode,
                "computed": psm is not None,
                "n_observations": len(self.dataset),
                "automatic_size_limit": self.config.psm_max_n,
                "path": "co_clustering_matrix.npy" if psm is not None else None,
                "reason": (
                    "requested_full"
                    if self.config.psm_mode == "full"
                    else "required_by_dahl"
                    if self.config.posterior_summary == "dahl"
                    else "within_automatic_size_limit"
                    if psm is not None
                    else "disabled"
                    if self.config.psm_mode == "none"
                    else "skipped_above_automatic_size_limit"
                ),
            },
        )
        chain_ids = np.concatenate(
            [
                np.full(draw_count, chain_index, dtype=np.int64)
                for chain_index, draw_count in enumerate(draws_per_chain)
            ]
        )
        self._save_diagnostic_array("partition_sample_chain_ids.npy", chain_ids)

        production_sses = compute_partition_sses(
            final_z,
            torch.as_tensor(samples, dtype=torch.long, device=final_z.device),
            self.config.n_clusters,
        ).cpu().numpy()
        # Summarize the production draws without feeding this estimate back
        # into the sampler; its eta remains the last joint-training value.
        final_eta = update_eta_posterior(
            n=len(self.dataset),
            p=int(self.dataset.x.shape[1]),
            d=self.config.z_dim,
            eta_a=self.config.eta_a,
            eta_b=self.config.eta_b,
            eta_shape_scale=self.config.eta_shape_scale,
            eta_multiplier=self.config.eta_multiplier,
            lambda_reconstruction=self.lambda_reconstruction,
            full_reconstruction_sse=final_reconstruction_sse,
            expected_cluster_sse=float(production_sses.mean()),
        )
        representative, summary, representative_filename = self._posterior_summary(
            samples, psm
        )
        metrics = self._metrics(representative)

        torch.save(self.model.state_dict(), self.output_dir / "model_state.pt")
        np.save(self.output_dir / "latent_final.npy", final_z.cpu().numpy())
        self._save_diagnostic_array(
            "persistent_partitions_final.npy",
            partitions.cpu().numpy(),
        )
        if self.partition_history:
            np.save(
                self.output_dir / "persistent_partition_history.npy",
                np.stack(self.partition_history),
            )
        save_json(self.output_dir / "eta_sampling.json", asdict(sampling_eta))
        save_json(self.output_dir / "eta_final.json", asdict(sampling_eta))
        self._save_diagnostic_json(
            "eta_production_diagnostic.json",
            {"role": "post-hoc recalculation only; not used in sampling", **asdict(final_eta)},
        )
        self._save_diagnostic_json("final_metrics.json", metrics)
        sampling_manifest = {
            "sampling_device": "cpu",
            "n_chains": self.config.n_chains,
            "partition_sampler": self.config.partition_sampler,
            "joint_mc_draws_per_iteration": self.config.mc_partitions,
            "joint_sweeps_per_chain_per_iteration": self.config.gibbs_sweeps_per_outer,
            "chain_initial_seeds": self.chain_seeds,
            "burnin_sweeps_per_chain": self.config.posterior_burnin_sweeps,
            "thinning": self.config.posterior_thinning,
            "draws_per_chain": draws_per_chain,
            "total_retained_draws": int(samples.shape[0]),
            "eta_sampling_mean": sampling_eta.mean,
            "eta_multiplier": sampling_eta.eta_multiplier,
            "eta_sampling_effective_mean": sampling_eta.effective_mean,
            "eta_source": "last_joint_iteration",
            "gibbs_precision": sampling_eta.effective_mean,
            "lambda_reconstruction_absent_from_allocation": True,
            "psm_computed": psm is not None,
            "sample_order": "chain-major",
        }
        self._save_diagnostic_json("posterior_sampling_manifest.json", sampling_manifest)

        runtime = float(time.time() - start)
        finished_utc = datetime.now(timezone.utc).isoformat()
        self.environment.update(
            {"finished_utc": finished_utc, "runtime_seconds": runtime}
        )
        self._save_diagnostic_json("environment.json", self.environment)
        result: Dict[str, object] = {
            "metrics": metrics,
            "posterior_summary": summary.to_dict(),
            "representative_partition": representative_filename,
            "lambda_reconstruction": self.lambda_reconstruction,
            "lambda_calibration": calibration if self.config.save_diagnostics else {
                key: calibration[key] for key in (
                    "target_gradient_ratio_r0", "gradient_ratio_median", "calibration_batches"
                ) if key in calibration
            },
            "eta": asdict(sampling_eta),
            "sampling_eta": asdict(sampling_eta),
            "production_eta_diagnostic": asdict(final_eta),
            "centroid_refresh_count": self.centroid_refresh_count,
            "centroid_refresh_policy": "before every neural minibatch, at current parameters",
            "pretrain_gradient_steps": self.pretrain_gradient_steps,
            "joint_gradient_steps": self.joint_gradient_steps,
            "methodology_variants": self._methodology_variants(),
            "gb_initialization": dict(self.initialization_manifest),
            "latent_scale_projection": self._scale_definition(),
            "latent_scale_projection_count": len(self.scale_history),
            "latent_scale_final_projection": self.scale_history[-1] if self.scale_history else None,
            "output_dir": str(self.output_dir),
            "source_combined_sha256": self.source_manifest["combined_sha256"],
            "runtime_seconds": runtime,
            "started_utc": self.started_utc,
            "finished_utc": finished_utc,
            "pretrained_kmeans_metrics": self.pretrain_metrics,
            "sampling": sampling_manifest,
            "environment": self.environment,
        }
        save_json(self.output_dir / "run_summary.json", result)
        self._log(" | ".join([
            f"[Complete] {summary.method}",
            *(f"{name.upper()}={value:.4f}" for name, value in metrics.items()),
            f"expected_VI={summary.empirical_expected_vi:.6g}", f"time={runtime:.1f}s",
        ]))
        self._log(f"[Output] {self.output_dir}")
        return result
