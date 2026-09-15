# ViNED: Visual–Neural Encoding and Decoding

ViNED is Frey Bence's MSc research project exploring the relationship between visual stimuli and neural activity recorded with implanted Neuropixels electrodes. It builds on **NEDS (Neural Encoding and Decoding at Scale)** and uses International Brain Laboratory (IBL) recordings.

## Research goal

Learn both directions of the visual–neural relationship:

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
5. Training and evaluating visual–neural predictions.

**Status:** this is an experimental research implementation. Temporal alignment, checkpoint restoration, and evaluation task routing need validation before interpreting model scores. See [AGENT.md](AGENT.md) for the findings and development priorities.

## Planned work

- Establish a reproducible visual–neural baseline.
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

The requirements describe this repository's visual–neural workflow:

| Area | Python dependencies |
| --- | --- |
| Model, metrics, training and tracking | Torch 2.2.1, Transformers 4.38.2, Accelerate 0.27.2, TorchEval 0.0.7, einops, Ray 2.10.0, wandb |
| Numerical processing and plots | NumPy 1.26.4, pandas, SciPy, scikit-learn, Matplotlib, tqdm, PyYAML |
| IBL sessions and processed spikes | ONE-api, ibllib/Brainbox, iblatlas, iblutil |
| Replay videos and CLIP images | opencv-python-headless 4.10.0.84, Pillow |
| Dataset storage and downloads | Datasets 2.17.1, PyArrow 14.0.2, huggingface_hub |
| Optional raw LFP processing | Additional dependencies in [requirements-lfp.txt](requirements-lfp.txt) |

[requirements.txt](requirements.txt) declares the core dependencies;
`constraints-windows-py310.txt` and `constraints-linux-py310.txt` pin the resolved
versions for each platform. Constraints alone do not install packages.

### Install after cloning

Run these commands from the cloned `vined` directory. These examples select CPU
Torch; for an NVIDIA GPU, substitute `requirements-torch-cu118.txt` for
`requirements-torch-cpu.txt`.

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-bootstrap.txt
.venv\Scripts\python.exe -m pip install --no-deps -r requirements-torch-cpu.txt
.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-windows-py310.txt
.venv\Scripts\python.exe -B script/check_environment.py --device cpu
```

Linux Bash:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-bootstrap.txt
.venv/bin/python -m pip install --no-deps -r requirements-torch-cpu.txt
.venv/bin/python -m pip install -r requirements.txt -c constraints-linux-py310.txt
.venv/bin/python -B script/check_environment.py --device cpu
```

The checker runs `pip check`, imports, CLI help, video/dataset round trips, and a
small synthetic model step without downloading research assets. For CUDA, also
run it with `--device cuda`. Existing editable installations and optional LFP
setup are covered in [environment setup and workflow commands](docs/environment.md).
See [validation results and blockers](docs/environment-validation.md) for the
limits of these checks.

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
account/project or set `WANDB_MODE=offline` for local runs. Follow the
[workflow commands](docs/environment.md#paths-and-workflow) for each stage.

## Research documents

- [IBL visual data specifications and parameters](docs/ibl-visual-data-specs.md) — sourced stimulus reference, replay settings, CLIP schema, and alignment requirements.
- [Hungarian research report](docs/Frey_Bence_ITLNUH_Beszámoló.pdf)
- [Hungarian presentation](docs/Frey_Bence_ITLNUH_prezentáció.pptx)

## Origin and attribution

ViNED derives from **[NEDS: Neural Encoding and Decoding at Scale](https://github.com/yzhang511/NEDS)** by Yizi Zhang and collaborators. The ViNED repository is **[FreyBence/vined](https://github.com/FreyBence/vined)**. The [archived NEDS README](docs/README_NEDS.md) preserves the earlier project description, schematic, usage instructions, and paper citation, including the local Bash command edits that predated this rewrite. It is historical documentation and is deprecated as a guide to ViNED.

Refer to that archive for the upstream citation. Preserve NEDS attribution when using or extending its work.

## Licensing

The inherited NEDS code and documentation retain their original **MIT license**, including `Copyright (c) 2024 Yizi Zhang`, reproduced unchanged in [LICENSE](LICENSE).

Frey Bence grants **no additional license** for his original ViNED contributions. Public availability does not grant permission to reuse, modify, or redistribute those contributions beyond rights provided by applicable law or GitHub's terms. The upstream MIT license continues to apply to inherited NEDS material; it does not serve as a blanket license for ViNED's additions.

See [LICENSING.md](LICENSING.md) for the scope and Git history reference.
