# Python environment and execution

ViNED is a **checkout-only application** using standalone **Python 3.10**,
**venv**, and **pip**. Install the dependencies of this repository and execute
its `src/*.py` entry scripts. The checkout supplies Python modules, YAML
configuration, and `data/*.txt` session lists; keep those paths together.
There is no wheel or editable package to install, and no NEDS checkout or
distribution is required. Python itself, NVIDIA drivers, and Linux Slurm are
external prerequisites. See the [README prerequisites](../README.md#prerequisites).

## Install

For an existing environment with an editable `vined` or `neds` installation,
uninstall that distribution using its venv Python (`python -m pip uninstall
vined neds`), then install the requirements below without an editable-install
step. Recreating the venv is recommended when trimming old dependencies:
installing a requirements file does not remove packages from earlier setups.
Python module import names are unchanged.

Run from the checkout root. Use a working 64-bit Python 3.10 interpreter explicitly;
do not assume `python` on PATH is the right version. Windows was created with
Python 3.10.11. An access-denied error from the Store Python in a restricted
execution environment may require running these commands in a normal terminal.

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-bootstrap.txt
.venv\Scripts\python.exe -m pip install --no-deps -r requirements-torch-cu118.txt
.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-windows-py310.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -B script/check_environment.py --device cpu
.venv\Scripts\python.exe -B script/check_environment.py --device cuda
```

Linux Bash:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-bootstrap.txt
.venv/bin/python -m pip install --no-deps -r requirements-torch-cu118.txt
.venv/bin/python -m pip install -r requirements.txt -c constraints-linux-py310.txt
.venv/bin/python -m pip check
.venv/bin/python -B script/check_environment.py --device cpu
.venv/bin/python -B script/check_environment.py --device cuda
```

For a CPU environment, replace `requirements-torch-cu118.txt` with
`requirements-torch-cpu.txt` and omit the CUDA check. Use a separate environment
when comparing CPU and CUDA wheels. The general requirements preserve Torch
versions without changing the installed CPU/CUDA variant. Always install Torch
first. ViNED's CLIP path uses Transformers and Pillow; neither `torchvision`,
`timm`, nor `torchaudio` is required. Seaborn is an IBL transitive dependency,
rather than a direct requirement of ViNED's source.

PyTorch 2.2.1 CUDA 11.8 wheels supply runtime libraries; a
separate CUDA toolkit is unnecessary for this repository's current code. An
NVIDIA driver and compatible GPU are still required. See the
[official wheel matrix](https://docs.pytorch.org/get-started/previous-versions/)
and [PyTorch explanation](https://discuss.pytorch.org/t/cuda-driver-cuda-toolkit-and-pytorch/198869).

In VS Code, select `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python`
(Linux). Activation is optional because all commands select Python explicitly.
Do not copy virtual environments between machines or operating systems.

### Optional LFP dependencies

The active spike/vision pipeline uses `requirements.txt`. The inherited raw LFP
branch additionally needs `requirements-lfp.txt` (SpikeInterface and explicit
Neuropixels/DSP dependencies). Install it only when using that branch:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-lfp.txt -c constraints-windows-py310.txt
.venv\Scripts\python.exe -B script/check_environment.py --lfp
```

On Linux use `.venv/bin/python` and `constraints-linux-py310.txt`. `ibllib` also
pulls in `ibl-neuropixel` for core data access; the optional file makes the LFP
module's direct dependencies explicit. The `--lfp` check verifies imports, not
raw recording processing or scientific validity.

## Paths and workflow

Run direct Python commands from the checkout root because model YAML includes
still use repository-relative paths. Bash wrappers change to that root themselves
and work when called from either the root or `script/`. Direct entry scripts
automatically put `src/` on Python's import path; Bash wrappers also export it
through `PYTHONPATH` for Ray/Slurm workers. Interactive Python sessions need the
checkout's `src/` on their import path explicitly.

| Override | Default | Used by |
| --- | --- | --- |
| `VENV_DIR` | `<checkout>/.venv` | Bash Python, Ray and distributed workers |
| `VINED_DATA_DIR` | `<checkout>/datasets` | Bash data paths and replay generator's ONE cache |
| `VINED_VISUAL_DIR` | `<data directory>/vis_stim` | Preparation and loader visual features; extraction wrapper output |
| `VINED_REPLAY_DIR` | `<checkout>/ibl_task_replay` | Replay generator and extraction wrapper input |
| `VINED_OUTPUT_DIR` | `<checkout>` | Bash training/evaluation outputs |
| `VINED_REPO_ROOT` | Discovered checkout | Override for Slurm submissions from another directory |

Relative path overrides are interpreted from the checkout. Prefer absolute
paths on clusters. Existing Python CLI arguments such as `--data_path`,
`--base_path`, `--video_dir`, and `--output_dir` take precedence for the entry
point receiving them; set `VINED_VISUAL_DIR` separately when moving features.

PowerShell examples (replace `EID` with a prepared session):

```powershell
$env:VINED_VISUAL_DIR = "$PWD/datasets/vis_stim"
.venv\Scripts\python.exe src/prepare_visual_stim.py --n_sessions 1 --eid EID --video_dir ./ibl_task_replay --output_dir ./datasets/vis_stim
.venv\Scripts\python.exe src/prepare_data.py --n_sessions 1 --eid EID --base_path ./datasets
.venv\Scripts\python.exe src/create_dataset.py --num_sessions 1 --eid EID --model_mode mm --mixed_training --base_path . --data_path ./datasets
.venv\Scripts\python.exe src/train.py --num_sessions 1 --eid EID --model_mode mm --mixed_training --base_path . --data_path ./datasets --config_dir ./src/configs --dummy_size 0
.venv\Scripts\python.exe src/eval.py --num_sessions 1 --eid EID --model_mode mm --mixed_training --base_path . --data_path ./datasets --mask_ratio 0.1 --enc_task_var random
```

Create replays first with `src/visual_stim_gen.py` after reviewing its session
list. It writes under `VINED_REPLAY_DIR`; it can download IBL data. CLIP extraction
downloads the model unless already cached. Preparation's inherited `--use_lfp`
flag enables the LFP branch despite its `store_false` implementation; the flag
semantics are unchanged in the current CLI.

Bash equivalents retain their original positional interfaces:

```bash
bash script/prepare_visual_stim.sh 1 EID
bash script/prepare_data.sh 1 EID
bash script/create_dataset.sh 1 EID
bash script/train.sh 1 EID train mm 0 0.1 False random
bash script/eval.sh 1 EID train mm 0.1 random False
```

`train.sh` search mode and `train_multi_gpu.sh` require Linux Slurm. Submit from
the checkout or `script/`, or export `VINED_REPO_ROOT` explicitly. Adjust inherited
`#SBATCH` account/partition/resources for your site. Every node needs the checkout,
data, and environment at the same absolute paths. For example, after site setup:

```bash
sbatch script/train_multi_gpu.sh 1 EID mm 0 0.1 random
```

The launcher runs `"$PYTHON" -m torch.distributed.run` on each worker. Ray workers
use the selected venv's executable. Windows single-GPU use is supported by the
setup; native Windows multi-node parity is not promised. See
[Ray platform limitations](https://docs.ray.io/en/latest/ray-overview/installation.html).

## Compatibility and validation

- NumPy remains 1.26.4; headless OpenCV is pinned to 4.10.0.84. Install only one OpenCV
  variant. Linux may need system OpenGL/GLib runtime libraries if `cv2` cannot
  import. Verify MP4 encoding/decoding instead of assuming a codec is available.
- Arrow is pinned to 14.0.2 for Ray 2.10.0 and Datasets 2.17.1. New Arrow releases
  remove legacy APIs; see the [upstream incompatibility](https://github.com/apache/arrow/issues/47155).
- IBL/SpikeInterface import and API compatibility must be checked. LFP-only
  imports are deferred until requested, so they do not prevent non-LFP preparation.
- The offline checker exercises dependency imports, CLI help, MP4 and `[T,768]`
  dataset round trips, path overrides, attention forward/backward, and a small
  full multimodal optimizer step and evaluation. Optional `--clip` checks cached
  CLIP weights; `--lfp` imports the optional LFP module;
  `--session-cache datasets/ibl_mm/train` uses two trusted prepared
  trials. It does not download data, run dummy GPU loads, or certify model quality.
- Full CLIP extraction, session training/evaluation, LFP data processing, and
  multi-node execution require their corresponding assets/infrastructure. Existing
  alignment, checkpoint restoration and routing findings remain research blockers.

See [environment validation](environment-validation.md) for actual results and
remaining blockers, including the clean checkout-only installation check.
