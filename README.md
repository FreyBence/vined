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

## Environment and execution

Use **Python 3.10 + pip + a local `.venv`** on Windows or Linux. The Python
distribution is `vined`. Install PyTorch first using the CPU or CUDA 11.8
requirements, then the general dependencies and editable package.

See [environment setup and workflow commands](docs/environment.md) and
[validation results and blockers](docs/environment-validation.md).

Bash wrappers resolve the checkout automatically and support `VENV_DIR` and
`VINED_*` path overrides. Linux Slurm account/partition settings remain
site-specific. Visual replay generation and CLIP extraction precede data preparation.

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
