# ViNED: Visualâ€“Neural Encoding and Decoding

ViNED is Frey Bence's MSc research project exploring the relationship between visual stimuli and neural activity recorded with implanted Neuropixels electrodes. It builds on **NEDS (Neural Encoding and Decoding at Scale)** and uses International Brain Laboratory (IBL) recordings.

## Research goal

Learn both directions of the visualâ€“neural relationship:

- **Encoding:** predict neural population activity from visual stimulus representations.
- **Decoding:** predict visual stimulus representations from neural activity.

The current model pairs spike counts with **CLIP embeddings** extracted from synthetic videos that replay the mouse's visual task. Its visual predictions are feature vectors; reconstructing images or playable videos would require an additional method.

## Current implementation

ViNED adapts the NEDS multimodal transformer, masking strategy, and session-specific projections. The active modalities are `spike` and `vision-clip`, replacing the original behavioral targets. Visual prediction uses cosine loss; neural prediction uses Poisson negative log-likelihood.

The data pipeline consists of:

1. Replaying trial stimuli from IBL task timing and wheel movements.
2. Extracting frame-level features with a frozen CLIP ViT-L/14 encoder.
3. Preparing visual features and binned neural activity for each trial.
4. Creating cached train, validation, and test datasets.
5. Training and evaluating visualâ€“neural predictions.

**Status:** this is an experimental research implementation. Temporal alignment, checkpoint restoration, and evaluation task routing need validation before interpreting model scores. See [AGENT.md](AGENT.md) for the findings and development priorities.

## Local data and outputs

Generated and downloaded artifacts default to the gitignored `output/` directory:

```text
output/
  datasets/          # ONE cache, aligned data, dataset caches, and vis_stim features
  visual_replays/    # replay videos and sidecars
  results/           # training checkpoints, evaluation metrics, and plots
  wandb/             # W&B run logs for training and evaluation
  eid-relevanc.txt   # session relevance discovery output
```

Published reproducibility inputs remain version-controlled under `data/`.
`VINED_OUTPUT_DIR` overrides the common output root. `VINED_DATA_DIR`,
`VINED_VISUAL_DIR`, and `VINED_REPLAY_DIR` override individual locations;
relative environment paths are resolved from the checkout root. Training,
fine-tuning, and evaluation also accept `--base_path` for their output root
and `--data_path` for datasets. Hyperparameter searches write `ray_results/`
under the output root. Explicit path overrides can point outside `output/`.

## Planned work

- Establish a reproducible visualâ€“neural baseline.
- Add task-event modalities to model relationships between stimuli, events, and neural responses.
- Define and evaluate prediction tasks for neural clusters and brain regions.
- Evaluate multi-session training and adaptation to held-out sessions.

These extensions are research objectives, not completed features.

## Repository guide

| Location | Purpose |
| --- | --- |
| [AGENT.md](AGENT.md) | Detailed project goal, code-review findings, and research roadmap |
| [docs/](docs/) | Hungarian research documents and project documentation |
| [data/](data/) | Session identifiers and training/evaluation session selections |
| [src/visual_stim_gen.py](src/visual_stim_gen.py) | Synthetic stimulus replay generation |
| [src/prepare_visual_stim.py](src/prepare_visual_stim.py) | CLIP feature extraction |
| [src/prepare_data.py](src/prepare_data.py) | IBL data preparation |
| [src/create_dataset.py](src/create_dataset.py) | Cached dataset creation |
| [src/multi_modal/](src/multi_modal/) | Multimodal transformer and embeddings |
| [src/train.py](src/train.py), [src/finetune.py](src/finetune.py) | Training and session adaptation |
| [src/eval.py](src/eval.py) | Evaluation entry point |
| [script/](script/) | Environment-specific Bash wrappers |

## Requirements and installation

ViNED is a **checkout-only research application**. Clone this repository, install
its dependencies, and run its scripts from the repository root. Keep `src/`,
`src/configs/`, and the session lists in `data/` together. No NEDS repository or
Python package is needed. ViNED does not provide a wheel, package installation,
or editable-install step.

### Prerequisites

- Git and **64-bit Python 3.10** with `venv` and pip, on Windows or Linux.
- A local `.venv` and disk space for IBL recordings, replay videos, CLIP weights,
  prepared datasets, and checkpoints; usage depends on the sessions selected.
- Internet access for dependency installation and initial IBL/CLIP downloads.
- For GPU execution: an NVIDIA GPU and a driver compatible with the CUDA 11.8
  PyTorch wheel. CPU setup is available for preparation and smoke checks.
- Bash for the optional shell wrappers (Git Bash on Windows); Linux Slurm for
  the supplied cluster/search launchers.

The requirements describe this repository's visualâ€“neural workflow:

| Area | Python dependencies |
| --- | --- |
| Model, metrics, training and tracking | Torch 2.2.1, Transformers 4.38.2, Accelerate 0.27.2, TorchEval 0.0.7, einops, Ray 2.10.0, wandb |
| Numerical processing and plots | NumPy 1.26.4, pandas, SciPy, scikit-learn, Matplotlib, tqdm, PyYAML |
| IBL sessions and processed spikes | ONE-api, ibllib/Brainbox, iblatlas, iblutil |
| Replay videos and CLIP images | opencv-python-headless 4.10.0.84, Pillow |
| Dataset storage and downloads | Datasets 2.17.1, PyArrow 14.0.2, huggingface_hub |
| Optional raw LFP processing | Additional dependencies in [requirements/lfp.txt](requirements/lfp.txt) |

[requirements/core.txt](requirements/core.txt) declares the core dependencies;
`requirements/constraints-windows-py310.txt` and `requirements/constraints-linux-py310.txt` pin the resolved
versions for each platform. Constraints alone do not install packages.

All dependency manifests live in `requirements/`. Keep `bootstrap.txt` separate
for installer tooling, `lfp.txt` for optional raw LFP processing, and the two
`torch-*.txt` files as alternative CPU/CUDA installs with their own package
indexes. The platform constraints preserve independently resolved environments;
they are not additional lists of packages to install.

### Install after cloning

Run these commands from the cloned `vined` directory. These examples select CPU
Torch; for an NVIDIA GPU, substitute `requirements/torch-cu118.txt` for
`requirements/torch-cpu.txt`.

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements/bootstrap.txt
.venv\Scripts\python.exe -m pip install --no-deps -r requirements/torch-cpu.txt
.venv\Scripts\python.exe -m pip install -r requirements/core.txt -c requirements/constraints-windows-py310.txt
```

Linux Bash:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements/bootstrap.txt
.venv/bin/python -m pip install --no-deps -r requirements/torch-cpu.txt
.venv/bin/python -m pip install -r requirements/core.txt -c requirements/constraints-linux-py310.txt
```

## Execution

Run entry points as `.venv/bin/python src/<script>.py` on Linux or
`.venv\Scripts\python.exe src/<script>.py` on Windows, from the checkout root.
Visual replay generation and CLIP extraction precede data preparation. Session
recordings, visual features, prepared caches, and trained checkpoints must be
obtained or generated for the workflow being run; cloning supplies source,
configuration, and session selections.

Bash wrappers resolve the checkout automatically and support `VENV_DIR` and
`VINED_*` path overrides. Linux Slurm account/partition settings remain
site-specific. Training enables W&B logging by default; configure your own
account/project or set `WANDB_MODE=offline` for local runs.

Wrapper arguments (run from the checkout; `--help` shows usage for the positional
training and evaluation launchers):

```text
bash script/create_dataset.sh [--eid EID | --eids-file FILE] [--n-sessions COUNT] [--alignment-root DIR] --split-strategy within_session --split-ratios 0.7 0.1 0.2 --split-seed 42
bash script/prepare_data.sh [--eid EID | --eids-file FILE] [--n-sessions COUNT]
bash script/prepare_visual_stim.sh [--eid EID] [--replay-dir output/visual_replays/RUN] [--sample-fps 5]
bash script/generate_replay.sh [--eid EID] [--projection on|off] [--force-reload]
bash script/train.sh COUNT EID TRAIN_MODE MODEL_MODE MASK_RATIO SEARCH TASK_VAR
bash script/eval.sh COUNT EID TRAIN_MODE MODEL_MODE MASK_RATIO TASK_VAR SEARCH [--overwrite]
bash script/train_multi_gpu.sh COUNT EID MODEL_MODE MASK_RATIO TASK_VAR
```

`generate_replay.sh` generates compressed lossless frames and replay metadata for all
EIDs in `data/eids.txt` by default; `--eid EID` selects one session. Every session
uses all trials. `--projection off` selects display-only output; projection is
on by default. It uses the explicit approximation settings in
`data/replay-config.json` (800×600, 30 Hz). Each run creates a new directory under
`VINED_REPLAY_DIR`, with one subdirectory per EID; `--output DIR` selects a specific new directory.
The wrapper always permits remote access, reusing cached files and downloading
missing data. Add `--force-reload` to bypass cached metadata responses and
re-download requested replay source datasets even when cached. Unrelated cache
contents and previous generated outputs are preserved.
Metadata/catalog lookup may still contact IBL. Incomplete
trials retain their outcomes. This pipeline does not generate MP4 videos. See the
[replay interface](docs/visual-replay/interface.md) for configuration and statuses.
Generation shows progress for all selected EIDs and for the current session's
trials. Trial progress starts after source preparation and includes unsuccessful
trial outcomes; it advances after artifact publication.

`prepare_visual_stim.sh` processes all published replay sessions by default;
`--eid EID` selects one session. Discovery uses the replay directory, not
`data/eids.txt`. It reads saved mouse-view replay images through verified
replay readback, without MP4 decoding. It preserves the full image extent for
CLIP and selects all observations unless `--sample-fps` is supplied. Select a
specific replay run when multiple generations exist. Outputs are compressed
`<eid>_visual_clip.npz` feature generations under `VINED_VISUAL_DIR`; existing
files are refused, so use a fresh `--output-dir` for another extraction.
Read them with `visual_features.FeatureArtifactReader`. These archives retain
coverage, trial outcomes, and generation identities; the legacy alignment loader
does not support them yet. See the [visual-features interface](docs/visual-features/interface.md).

`TRAIN_MODE` is `train` or `finetune`; `MODEL_MODE` is `mm`, `encoding`, or
`decoding`; `SEARCH` is `True` or `False`; `TASK_VAR` is `all`, `random`, or
`vision-clip`. `COUNT` is positive, `MASK_RATIO` is between 0 and 1, and `EID`
is a session UUID (or `None` for multi-session selection). Fine-tuning still
requires an actual EID. The ineffective `dummy_size` positional argument has
been removed from both training wrappers. Evaluation overwrites only when
`--overwrite` is supplied. Use the Python entry points directly for additional
options such as `--pretrain_task_var`.

Slurm is a shared-cluster job scheduler. `sbatch script/...` applies the
`#SBATCH` resource requests; ordinary Bash execution ignores them. Search and
multi-node training currently require Slurm; ordinary single-process training
does not. Scheduler accounts and partitions must match your cluster.

## Research documents

- [IBL visual data specifications and parameters](docs/ibl-visual-data-specs.md) â€” sourced stimulus reference, replay settings, CLIP schema, and alignment requirements.
- [Hungarian research report](docs/Frey_Bence_ITLNUH_BeszĂˇmolĂł.pdf)
- [Hungarian presentation](docs/Frey_Bence_ITLNUH_prezentĂˇciĂł.pptx)

## Origin and attribution

ViNED derives from **[NEDS: Neural Encoding and Decoding at Scale](https://github.com/yzhang511/NEDS)** by Yizi Zhang and collaborators. The ViNED repository is **[FreyBence/vined](https://github.com/FreyBence/vined)**. The [archived NEDS README](docs/README_NEDS.md) preserves the earlier project description, schematic, usage instructions, and paper citation, including the local Bash command edits that predated this rewrite. It is historical documentation and is deprecated as a guide to ViNED.

Refer to that archive for the upstream citation. Preserve NEDS attribution when using or extending its work.

## Licensing

The inherited NEDS code and documentation retain their original **MIT license**, including `Copyright (c) 2024 Yizi Zhang`, reproduced unchanged in [LICENSE](LICENSE).

Frey Bence grants **no additional license** for his original ViNED contributions. Public availability does not grant permission to reuse, modify, or redistribute those contributions beyond rights provided by applicable law or GitHub's terms. The upstream MIT license continues to apply to inherited NEDS material; it does not serve as a blanket license for ViNED's additions.

See [LICENSING.md](LICENSING.md) for the scope and Git history reference.
