# pip/venv migration validation

The migration and rename sections below are historical records. The current
installation contract is **checkout-only**; see the
[2026-09-16 validation](#checkout-only-validation-2026-09-16) and current
[setup instructions](environment.md). Editable installation is no longer used.

Validation performed on 2026-09-15. No Conda-exclusive code dependency was found.
The repository now uses standalone Python 3.10, pip requirements, and `.venv`.
The existing machine-wide Conda installation was not removed.

## Environments and recorded versions

| Platform | Environment | Version record |
| --- | --- | --- |
| Windows x86-64 | Python 3.10.11; Torch 2.2.1+cu118; torchvision 0.17.1+cu118; RTX 3060 Laptop GPU, driver 595.97 | [Windows constraints](../constraints-windows-py310.txt) |
| Linux x86-64 | Python 3.10.21 in the official `python:3.10-slim` container; standalone venv; Torch 2.2.1+cpu and torchvision 0.17.1+cpu | [Linux constraints](../constraints-linux-py310.txt) |

The Linux image used digest
`sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2`.
Both environments preserve NumPy 1.26.4, Datasets 2.17.1, Transformers 4.38.2,
Accelerate 0.27.2, TorchEval 0.0.7, and timm 0.9.16. New compatibility pins include
Ray 2.10.0, Arrow 14.0.2, and headless OpenCV 4.10.0.84.

The platform differences are Qt 5.15.2 on Windows versus 5.15.19 on Linux, and
`hf-xet` on Linux. Torch's local CPU/CUDA version suffix is intentionally omitted
from constraints; the dedicated wheel requirements select it. Linux constraints
were derived from a CPU environment; Linux CUDA-specific dependencies are supplied
by the selected Torch wheel's dependency declarations and have not been runtime-tested.

## Checks

| Check | Result |
| --- | --- |
| Windows `pip check` and requirements/constraints resolution | Passed |
| Linux clean pip installation and `pip check` | Passed |
| Second clean Linux installation from frozen constraints, including editable package installation | Passed |
| Imports including Ray Tune, IBL, SpikeInterface/LFP utilities and project modules | Passed on Windows and Linux |
| CLI help for training, fine-tuning, evaluation, data preparation, dataset creation and visual extraction | Passed on Windows and Linux |
| Tiny MP4 write/read, with all three frames decoded | Passed on Windows and Linux |
| Datasets/Arrow round trip with `[4,768]` visual arrays | Passed on Windows and Linux |
| Repository-relative visual override and absolute YAML includes | Passed in environment checks |
| Project attention forward/backward | Passed on Windows CPU/CUDA and Linux CPU |
| Small full multimodal optimizer step and evaluation | Passed on Windows CPU and Linux CPU using synthetic data |
| Prepared-session GPU step and evaluation | Passed on two cached trials from `0c828385-6dd6-4842-a702-c5075f5f5e81`, with `[100,399]` spikes and `[100,768]` vision |
| Cached CLIP extraction | Passed on Windows CUDA using `openai/clip-vit-large-patch14`, yielding finite `[1,768]` features without downloading |
| Bash syntax, root/script invocation, paths with spaces and simulated spooled Slurm worker arguments | Passed under Git Bash and native Linux Bash |

The prepared-session check uses a small newly initialized model and an in-memory
optimizer step. It does not run the production multi-epoch training/evaluation
workflow or restore an existing research checkpoint. No datasets or checkpoints
were overwritten. The CLIP check uses a synthetic image with the cached model.

## Blocker register

| Status | Finding | Resolution or remaining action |
| --- | --- | --- |
| **Resolved** | Python 3.10 reported access denied in the sandbox. | The registered Python 3.10.11 works in a normal execution context and created `.venv`. |
| **Resolved** | Bundled pip 23.0.1 rejected wheel metadata and attempted unavailable build dependencies on the Torch index. | Bootstrap pip 26.2.1; install only Torch/torchvision with `--no-deps` from their index, then install general dependencies from PyPI. |
| **Resolved** | Ray 2.9.3 Tune metadata conflicted with Arrow 14. | Ray 2.10.0 resolves and imports successfully with Arrow 14.0.2. |
| **Resolved** | IBL pulled headless OpenCV alongside the initially declared GUI variant. | Declare only `opencv-python-headless==4.10.0.84`; both platform codec checks pass. |
| **Resolved** | Conda activation, working-directory assumptions and machine-specific paths blocked portable execution. | Explicit venv interpreter, shared path defaults/overrides, and argument-preserving distributed launch. |
| **Resolved** | Unconditional LFP imports could block non-LFP preparation. | Defer them to the existing LFP branch; imports also pass in the validated environments. |
| **Non-blocking platform limitation** | Windows Torch reports no compiled flash attention. | CUDA forward/backward succeeds through the available SDPA implementation. Expect possible performance differences; Conda removal does not require flash attention. |
| **External prerequisite** | Python, NVIDIA driver and Slurm cannot be installed as ordinary pip project dependencies. | Install/provision them separately. Python and Windows GPU are verified here. |
| **Pending infrastructure validation** | Linux CUDA and real Slurm/Ray multi-node jobs were not executed. | Run GPU checks and a short cluster job on the target allocation. Mocked argument checks do not establish distributed correctness. |
| **Pending workflow validation** | Full IBL download/preparation, raw LFP processing, production training/checkpoint restoration/evaluation and online W&B logging were not exercised end to end. | Validate on selected sessions before retiring the old environment used for experiments. |
| **Existing research blockers** | Alignment, trial validity, checkpoint restoration and task routing concerns remain. | Follow AGENT.md's research validation priorities; passing environment checks does not establish scientific correctness. |

## Reproduction

Follow [environment.md](environment.md). The reusable checks are:

```powershell
.venv\Scripts\python.exe -B script/check_environment.py --device cpu
.venv\Scripts\python.exe -B script/check_environment.py --device cuda --clip --session-cache datasets/ibl_mm/train
python script/check_launchers.py --bash "C:/Program Files/Git/bin/bash.exe"
```

On Linux use `.venv/bin/python` and `--bash bash`. `--clip` requires cached model
weights; `--session-cache` reads trusted project-generated pickle-backed `.npy`
files. Omit those flags for offline synthetic checks without research assets.

To refresh constraints after an intentional dependency change, first install and
validate in a fresh environment, then run `script/freeze_environment.py` with the
appropriate platform constraint filename. Preserve the opposite platform file;
do not copy a Windows package inventory into a Linux lock.

## Checkout rename validation (2026-09-15)

After renaming the Windows checkout to vined, the local .venv was recreated
with Python 3.10.11, the pinned Windows dependencies, and CUDA 11.8 wheels.
The editable vined distribution and pip executable resolve to the new checkout.
Dependency checks, all CLI checks, CPU synthetic model checks, CUDA prepared-session
checks, cached CLIP extraction, and Git Bash launcher checks passed. The old
environment backup was removed after validation. No stale old-checkout references
were found in the checked environment variables, shell profiles, or Desktop/Start
Menu shortcuts. The project naming notes in AGENT.md were updated.

## Checkout-only validation (2026-09-16)

ViNED now runs directly from its checkout. `src/setup.py` and
`src/pyproject.toml` have been removed; dependencies are installed from the
requirements files with platform constraints. No NEDS checkout or installed
project distribution is needed. The checkout retains Python modules, YAML
configuration and session lists together.

The core requirements no longer install timm, torchvision or SpikeInterface.
Seaborn remains a dependency of ibllib. Optional raw LFP processing uses
`requirements-lfp.txt`; `script/check_environment.py --lfp` checks its imports.
The platform constraints retain optional/transitive dependencies and omit the
obsolete timm/torchvision pins. The inventory helper excludes both old local
distribution names (`neds` and `vined`).

| Check | Result |
| --- | --- |
| Fresh Linux Python 3.10.21 venv in `python:3.10-slim`, CPU Torch, revised core requirements and Linux constraints | Passed, with no editable/package install |
| Installed-distribution inventory before optional LFP installation | Confirmed absence of vined, neds, timm, torchvision and SpikeInterface |
| Linux `pip check`, core/project imports and all six entry-point help commands | Passed |
| Linux MP4 round trip, `[T,768]` dataset round trip, path/config checks, attention gradients and synthetic multimodal optimizer/evaluation step | Passed |
| Add optional LFP requirements with the same constraints, then `pip check` and LFP imports | Passed |
| Frozen Linux inventory after adding LFP | Matches the previous pins except for removed timm/torchvision |
| Existing Windows Python 3.10.11 / CUDA 11.8 venv: updated checker with `--device cuda --clip --lfp` | Passed, including cached CLIP feature extraction and all six CLI checks |
| Native Linux Bash and Windows Git Bash launcher checks | Passed, including paths with spaces, preserved `PYTHONPATH`, checkout source exposure and simulated Slurm worker arguments |
| Freeze helper against the existing Windows environment | Excludes the legacy editable vined distribution and preserves Torch variant-independent pins |

The Windows check reused the existing environment, which still contains packages
from the earlier setup. The fresh Linux environment establishes independence
from those packages and from an editable install. No existing Windows environment,
research data, or checkpoint was removed. Full production training, raw LFP
processing and real cluster execution were not rerun; the research blockers
listed above still apply.
