# GB Cluster–Joint

GB Cluster–Joint is a Python framework for generalized Bayesian clustering with autoencoder-based representation learning. It combines reconstruction error and within-cluster dispersion in a loss-based formulation, alternating between neural network optimization and Gibbs updates of cluster allocations. After training, posterior partition samples conditional on the fitted representation are used to obtain a representative clustering and summarize uncertainty through posterior co-clustering probabilities.


## Installation

Use Python 3.10 or later. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The Simulation I configuration also requires R, with `Rscript` available on `PATH`, and these packages:

```r
install.packages(c("salso", "RcppCNPy", "jsonlite", "digest"))
```

Hartigan–Wong initialization uses R's `stats::kmeans`; the final VI estimate uses SALSO. The toy example below uses a VI medoid and runs without R.

## Quick Demo

```sh
python examples/generate_toy_data.py --output-dir data/toy --seed 123
python train.py \
  --preset presets/smoke.json \
  --x-path data/toy/X.npy --y-path data/toy/y.npy \
  --output-dir outputs/smoke --n-clusters 3 --seed 125 --device cpu
```

This generates 90 observations and runs two pretraining epochs and two joint iterations. It checks the training and output paths; its data generator and short training budget are not those used for the manuscript simulations.

## Simulation I configuration

Place the preprocessed feature matrix in `data/simulation1/X.npy` and, optionally, reference labels in `data/simulation1/y.npy`. Initialize on the same matrix, then run training:

```sh
python scripts/initialize_hartigan.py \
  --x-path data/simulation1/X.npy \
  --output-dir outputs/hw_seed125 \
  --n-clusters 4 --seed 125 --n-starts 100

python train.py \
  --preset presets/simulation1_current.json \
  --x-path data/simulation1/X.npy --y-path data/simulation1/y.npy \
  --initial-partition-path outputs/hw_seed125/labels.npy \
  --output-dir outputs/joint_seed125 \
  --n-clusters 4 --seed 125 --device cpu
```

Omit `--y-path` when reference labels are unavailable. They are used only for evaluation and must not be supplied as the initial partition. Use a new output directory for each run; existing output directories are not overwritten.

The supplied preset specifies:

| Setting | Value |
|---|---|
| Autoencoder | Depend on the Settings (Linear `p → 6 → p` with default) |
| Optimizer | Adam |
| Learning rates | Pretraining `0.005`; joint training `0.001` |
| Neural minibatch size | 64 observations |
| Training budget | 100 pretraining epochs; 100 joint iterations |
| Neural work per iteration | One full minibatch epoch |
| Initialization | Provided raw-feature Hartigan–Wong partition, 100 starts |
| Reconstruction weight | Calibrated in pretraining with target gradient ratio r_0 = 1; then fixed |
| Training Gibbs sampler |  `B=10` sweeps and `M=5` retained partitions per iteration |
| Latent scale projection | After pretraining and each joint epoch |
| Eta | Shape–rate Gamma update; prior shape and rate 1; multiplier 1 |
| Gaussian prior precision | `1e-4` for encoder and decoder |
| Final sampling | 100 burn-in sweeps; 500 retained partitions; |
| Final estimate | SALSO with VI loss |

`B` counts full Gibbs sweeps, `M` counts retained training partitions, and the minibatch size counts observations per gradient step.  Final sampling is a separate stage after training.

Explicit command-line options override preset values. For the joint-learning-rate comparisons, add `--lr-joint 0.0005` or `--lr-joint 0.0002` and choose a separate output directory. Generic CLI defaults differ from this preset, including the architecture and calibration stage; use the preset to select the configuration above. Other options are listed by `python train.py --help`.

## Input and output files

`X.npy` must be a finite numeric array with observations in rows. Values are converted to float32. The optional `y.npy` and the initial partition each contain one label per input row. The initial partition must have exactly the requested number of nonempty clusters. Keep the feature matrix, labels, and initializer in the same row order.

The supplied Simulation I preset expects features already standardized using column means and sample standard deviations (`ddof=1`). The optional `--standardize zscore` instead uses population standard deviations (`ddof=0`). Regenerate the initializer whenever the input transformation changes.

The main outputs are:

| File | Contents |
|---|---|
| `config.json`, `run_summary.json` | Resolved settings and final summaries |
| `training_history.csv` | Iteration losses, eta, and evaluation metrics when available |
| `model_state.pt`, `latent_final.npy` | Fitted model and representation |
| `partition_samples.npy` | Final retained partitions |
| `representative_partition.npy`, `posterior_summary.json` | Selected estimate and decision settings |
| `eta_sampling.json` | Eta frozen for final sampling |
| `source_manifest.json`, `data_manifest.json`, `initial_partition_manifest.json` | Source, input, and initialization provenance |
| `execution.json` | Device and thread settings |

Add `--save-diagnostics` for intermediate checkpoints, calibration records, and chain histories. By default, `co_clustering_matrix.npy` is saved only for `n ≤ 2500`; `--psm-mode none` disables it and `--psm-mode full` requests it for larger inputs. Its storage grows quadratically with sample size.


