from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from gb_joint.architectures import SUPPORTED_ARCHITECTURES
from gb_joint.config import TrainConfig
from gb_joint.data import load_npy_dataset
from gb_joint.trainer import TotalLossTrainer
from gb_joint.utils import save_json, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Fit generalized Bayes clustering with an autoencoder: "
            "lambda_R times raw reconstruction SSE plus raw cluster SSE."
        )
    )
    parser.add_argument(
        "--preset",
        type=Path,
        default=None,
        help=(
            "Optional JSON file supplying method/architecture defaults. "
            "Explicit command-line options take precedence."
        ),
    )
    parser.add_argument("--x-path", type=Path, required=True)
    parser.add_argument(
        "--y-path",
        type=Path,
        default=None,
        help="Optional reference labels used only for post-fit evaluation.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--z-dim", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=None,
        help="Encoder hidden widths for architecture=mlp, e.g. 256 64; overrides --hidden-dim.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Must be zero for deterministic encoder and decoder maps.",
    )
    parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default="mlp_ln_gelu",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--outer-iterations", type=int, default=100)
    parser.add_argument("--joint-epochs-per-outer", type=int, default=1)
    parser.add_argument("--gibbs-sweeps-per-outer", type=int, default=5)
    parser.add_argument("--mc-partitions", type=int, default=5)
    parser.add_argument(
        "--partition-initialization", choices=["latent_kmeans", "provided"],
        default="latent_kmeans",
        help="Use pretrained latent K-means or a supplied initial partition.",
    )
    parser.add_argument(
        "--initial-partition-path", type=Path, default=None,
        help="NPY cluster assignments used as the initial partition, separate from --y-path.",
    )
    parser.add_argument(
        "--latent-scale-projection", action=argparse.BooleanOptionalAction,
        default=False,
        help="Project pooled latent variance to one (linear architecture on CPU only).",
    )
    parser.add_argument(
        "--partition-sampler",
        choices=["independent_chains", "single_chain_sequential"],
        default="single_chain_sequential",
    )
    parser.add_argument("--initial-chain-burnin-sweeps", type=int, default=0)

    parser.add_argument("--posterior-burnin-sweeps", type=int, default=100)
    parser.add_argument("--posterior-samples", type=int, default=100)
    parser.add_argument("--posterior-thinning", type=int, default=1)
    parser.add_argument(
        "--posterior-summary", choices=["salso", "vi_medoid", "dahl"], default="salso",
        help="Partition summary: SALSO VI search, VI medoid, or Dahl criterion.",
    )
    parser.add_argument("--salso-n-runs", type=int, default=64)
    parser.add_argument("--salso-n-cores", type=int, default=1)
    parser.add_argument(
        "--psm-mode",
        choices=["auto", "full", "none"],
        default="auto",
        help=(
            "Write the full n-by-n co-clustering matrix always, never, or only "
            "when n does not exceed --psm-max-n."
        ),
    )
    parser.add_argument("--psm-max-n", type=int, default=2500)

    parser.add_argument(
        "--target-gradient-ratio",
        type=float,
        default=1.0,
        help="r0 in median(lambda_R ||grad R|| / ||grad C||) = r0.",
    )
    parser.add_argument("--gradient-epsilon", type=float, default=1e-12)
    parser.add_argument("--calibration-batches", type=int, default=None)
    parser.add_argument(
        "--lambda-calibration-stage",
        choices=["before_warmup", "after_warmup"],
        default="after_warmup",
    )
    parser.add_argument(
        "--eta-initialization",
        choices=["prior_mean", "pretrained_loss"],
        default="pretrained_loss",
    )
    parser.add_argument("--eta-a", type=float, default=1.0)
    parser.add_argument("--eta-b", type=float, default=1.0)
    parser.add_argument(
        "--eta-shape-scale",
        type=float,
        default=1.0,
        help=(
            "Positive C in Gamma shape eta_a + C * n * (p + d) / 2; "
            "rate unchanged. C != 1 is a sensitivity variant."
        ),
    )
    parser.add_argument(
        "--eta-multiplier",
        type=float,
        default=1.0,
        help=(
            "Use this multiplier times the base Gamma mean for Gibbs and neural "
            "updates, including final sampling. The base Gamma update is unchanged; "
            "non-unit values are an algorithmic sensitivity variant."
        ),
    )
    parser.add_argument(
        "--optimizer", choices=["sgd", "adam"], default="adam",
        help=(
            "Neural optimizer. SGD uses no momentum and requires both learning rates."
        ),
    )
    for option, default_rate in (("--lr-pretrain", 5e-4), ("--lr-joint", 2e-4)):
        parser.add_argument(
            option, type=float, default=default_rate,
            help=(
                "Adam learning rate. For SGD, supply both rates through the CLI or preset."
            ),
        )
    parser.add_argument("--encoder-prior-precision", type=float, default=1e-4)
    parser.add_argument("--decoder-prior-precision", type=float, default=1e-4)
    parser.add_argument("--standardize", choices=["none", "zscore"], default="none")
    parser.add_argument("--seed", type=int, default=125)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--num-threads", type=int, default=1,
        help="CPU numerical-library threads (default: one for reproducible runs).",
    )
    parser.add_argument(
        "--save-diagnostics", action="store_true",
        help="Also save intermediate checkpoints, chain histories and detailed manifests.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, object]]:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--preset", type=Path, default=None)
    preliminary_args, _ = preliminary.parse_known_args(argv)

    parser = build_parser()
    preset_values: dict[str, object] = {}
    if preliminary_args.preset is not None:
        preset_path = preliminary_args.preset.expanduser().resolve()
        if not preset_path.is_file():
            parser.error(f"Preset file does not exist: {preset_path}")
        with preset_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            parser.error("A preset must be a JSON object.")
        allowed = {
            action.dest
            for action in parser._actions
            if action.dest not in {"help", "preset"}
        }
        forbidden = {"x_path", "y_path", "output_dir", "initial_partition_path"}
        unknown = set(loaded) - allowed
        path_fields = set(loaded) & forbidden
        if unknown:
            parser.error(
                "Unknown preset field(s): " + ", ".join(sorted(unknown))
            )
        if path_fields:
            parser.error(
                "Data and output paths must be supplied explicitly, not in a "
                "preset: " + ", ".join(sorted(path_fields))
            )
        preset_values = loaded
        parser.set_defaults(**preset_values)
    args = parser.parse_args(argv)
    if args.optimizer == "sgd":
        explicit_options = {
            arg.split("=", 1)[0] for arg in (sys.argv[1:] if argv is None else argv)
        }
        missing = [
            field for field in ("lr_pretrain", "lr_joint")
            if field not in preset_values
            and "--" + field.replace("_", "-") not in explicit_options
        ]
        if missing:
            parser.error(
                "SGD requires explicitly supplied learning rates: "
                + ", ".join("--" + field.replace("_", "-") for field in missing)
                + " (or the corresponding preset fields)."
            )
    return args, preset_values


def main(argv: list[str] | None = None) -> dict[str, object]:
    args, preset_values = parse_args(argv)
    if args.num_threads < 1:
        raise SystemExit("--num-threads must be positive.")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(
            f"Refusing to write into an existing output directory: {output_dir}"
        )
    config = TrainConfig(
        x_path=args.x_path,
        y_path=args.y_path,
        output_dir=output_dir,
        n_clusters=args.n_clusters,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        architecture=args.architecture,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        outer_iterations=args.outer_iterations,
        joint_epochs_per_outer=args.joint_epochs_per_outer,
        gibbs_sweeps_per_outer=args.gibbs_sweeps_per_outer,
        mc_partitions=args.mc_partitions,
        partition_initialization=args.partition_initialization,
        initial_partition_path=args.initial_partition_path,
        latent_scale_projection=args.latent_scale_projection,
        partition_sampler=args.partition_sampler,
        initial_chain_burnin_sweeps=args.initial_chain_burnin_sweeps,
        posterior_burnin_sweeps=args.posterior_burnin_sweeps,
        posterior_samples=args.posterior_samples,
        posterior_thinning=args.posterior_thinning,
        posterior_summary=args.posterior_summary,
        salso_n_runs=args.salso_n_runs,
        salso_n_cores=args.salso_n_cores,
        psm_mode=args.psm_mode,
        psm_max_n=args.psm_max_n,
        target_gradient_ratio=args.target_gradient_ratio,
        gradient_epsilon=args.gradient_epsilon,
        calibration_batches=args.calibration_batches,
        lambda_calibration_stage=args.lambda_calibration_stage,
        eta_initialization=args.eta_initialization,
        eta_a=args.eta_a,
        eta_b=args.eta_b,
        eta_shape_scale=args.eta_shape_scale,
        eta_multiplier=args.eta_multiplier,
        optimizer=args.optimizer,
        lr_pretrain=args.lr_pretrain,
        lr_joint=args.lr_joint,
        encoder_prior_precision=args.encoder_prior_precision,
        decoder_prior_precision=args.decoder_prior_precision,
        standardize=args.standardize,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        verbose=not args.quiet,
        save_diagnostics=args.save_diagnostics,
    )
    config.validate()
    dataset, metadata = load_npy_dataset(
        config.x_path, config.y_path, standardize=config.standardize
    )
    if config.n_clusters > len(dataset):
        raise SystemExit("n_clusters cannot exceed the number of observations.")
    TotalLossTrainer.validate_dataset_contract(config, dataset)
    TotalLossTrainer.validate_runtime_dependencies(config)

    output_dir.mkdir(parents=True, exist_ok=False)
    if config.save_diagnostics:
        save_json(
            output_dir / "command.json",
            {
                "argv": [sys.executable, *(
                    sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]
                )],
                "cwd": os.getcwd(),
                "invoked_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    if config.save_diagnostics and args.preset is not None:
        preset_path = args.preset.expanduser().resolve()
        save_json(
            output_dir / "preset_manifest.json",
            {
                "path": str(preset_path),
                "sha256": sha256_file(preset_path),
                "values": preset_values,
                "command_line_overrides_take_precedence": True,
            },
        )
    if config.save_diagnostics:
        for name, array in metadata.items():
            np.save(output_dir / f"{name}.npy", array)
    elif metadata:
        np.savez(output_dir / "preprocessing.npz", **metadata)

    torch.set_num_threads(args.num_threads)
    save_json(output_dir / "execution.json", {
        "cpu_threads": args.num_threads,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "numerical_libraries_limited_by_threadpoolctl": True,
    })
    with threadpool_limits(limits=args.num_threads):
        result = TotalLossTrainer(config, dataset).fit()
    return result


if __name__ == "__main__":
    torch.set_num_interop_threads(1)
    main()
