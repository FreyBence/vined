# ViNED technology stack

ViNED (Visual-Neural Encoding and Decoding) is a Python machine-learning research project derived from NEDS. It uses a custom multimodal transformer to learn relationships between neural spike counts and visual features from synthetic IBL task replay videos. Its current visual predictions are CLIP feature vectors, rather than reconstructed images or videos.

This summary reflects the repository's dependency declarations and source code. Core versions are declared in [requirements.txt](../requirements.txt); platform constraints pin the resolved environment. See the [validation report](environment-validation.md) for tested platforms and limitations.

## Language and environment

| Technology | Declared version | Explanation and project role |
| --- | --- | --- |
| Python | 3.10 (validated patch in environment report) | Main language for data preparation, model implementation, training, and evaluation. |
| venv | Python standard library | Creates a project-local `.venv` from standalone Python; see [setup](environment.md). |
| pip, setuptools and wheel | [Bootstrap pins](../requirements-bootstrap.txt) | Install third-party dependencies into `.venv`. ViNED runs directly from its checkout and has no Python distribution or editable-install step. |
| NVIDIA CUDA runtime | 11.8 | Runtime dependencies supplied by the CUDA PyTorch wheels; NVIDIA driver is external. No standalone toolkit is required by current code. |

## Models and training

| Technology | Declared version | Explanation and project role |
| --- | --- | --- |
| PyTorch | 2.2.1 | Supplies tensors, automatic differentiation, neural-network layers, data loaders, optimizers, and checkpoint saving. The custom multimodal transformer and session-specific projections are implemented with it. |
| Hugging Face Transformers | 4.38.2 | Loads the CLIP image model and processor, and supplies activation functions used by the custom model. |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` checkpoint | Frozen visual encoder that converts replay frames into normalized, 768-dimensional feature vectors for the `vision-clip` modality. |
| Hugging Face Accelerate | 0.27.2 | Coordinates device placement and distributed training through `Accelerator`. |
| Ray Tune | 2.10.0 (`ray[default,tune]`) | Runs optional hyperparameter searches; training scripts use the ASHA scheduler to manage trials. |
| einops | — | Expresses tensor reshaping and repetition used in attention and multimodal model code. |
| Weights & Biases (`wandb`) | — | Tracks experiment configuration and training/evaluation metrics. |

Sources: [model implementation](../src/multi_modal/), [trainer](../src/trainer/base.py), [training entry point](../src/train.py), and [visual feature extractor](../src/prepare_visual_stim.py).

## Scientific computing and evaluation

| Technology | Declared version | Explanation and project role |
| --- | --- | --- |
| NumPy | 1.26.4 | Handles numerical arrays, spike and visual data transformations, and compressed visual-feature archives. |
| pandas | — | Handles tabular trial information and recording metadata. |
| SciPy | — | Provides interpolation, sparse spike representations, signal processing, and mathematical functions for evaluation. |
| scikit-learn | — | Supplies data splitting, clustering, preprocessing, and metrics such as R-squared and balanced accuracy. Some utilities belong to inherited analysis paths. |
| TorchEval | 0.0.7 | Provides tensor-based R-squared evaluation through `R2Score`. |
| Matplotlib | — | Generates evaluation plots and scientific figures. |
| tqdm | — | Displays progress for preprocessing, feature extraction, and training loops. |

## Neuroscience data access and processing

| Technology | Declared version | Explanation and project role |
| --- | --- | --- |
| IBL ONE API (`ONE-api`) | — | Retrieves International Brain Laboratory sessions, trial data, and recording data using session identifiers. |
| ibllib / Brainbox | — | Loads session data and existing spike-sorting results, and provides spike binning and behavioral utilities. |
| iblatlas | — | Provides brain-region identifiers and anatomical mappings through `BrainRegions`. |
| iblutil | — | Provides IBL numerical helpers such as two-dimensional binning and membership matching. |
| ibl-neuropixel | Optional LFP requirement; also pulled in by ibllib | Supports Neuropixels recording formats and signal processing; the LFP utility imports `neuropixel`, `spikeglx`, and `ibldsp`. |
| SpikeInterface | Optional LFP requirement | Reads and preprocesses electrophysiology recordings in the LFP utility, including filtering, channel correction, and referencing. |

The main adaptation consumes processed spike counts. The repository also contains local field potential (LFP) preprocessing utilities; these do not constitute a new spike-sorting pipeline. See [IBL data utilities](../src/utils/ibl_data_utils.py) and [LFP preprocessing](../src/utils/preprocess_lfp.py).

## Visual processing and dataset storage

| Technology | Declared version | Explanation and project role |
| --- | --- | --- |
| OpenCV (`cv2`, headless wheel) | 4.10.0.84 | Creates synthetic replay videos and reads/samples their frames for visual feature extraction. |
| Pillow (`PIL`) | Explicitly declared | Converts video frames into image objects accepted by the CLIP processor. |
| Hugging Face Datasets | 2.17.1 | Represents trial datasets, combines sessions, and saves/loads prepared dataset splits. |
| Hugging Face Hub (`huggingface_hub`) | Explicitly declared | Supports dataset discovery and optional upload helpers; the code also supports local dataset loading. |
| Local files: NPZ, pickle, and PyTorch `.pt` | Not applicable | Store compressed CLIP features, serialized Python data, and model checkpoints. Prepared datasets and replay videos are also stored on disk. |

Sources: [replay generation](../src/visual_stim_gen.py), [feature extraction](../src/prepare_visual_stim.py), [dataset utilities](../src/utils/dataset_utils.py), and [dataset creation](../src/create_dataset.py).

## Configuration and execution

| Technology | Explanation and project role |
| --- | --- |
| YAML and PyYAML | Store and parse model/trainer configuration under `src/configs/`. The repository implements its own `DictConfig` wrapper; PyYAML is explicitly declared. |
| Python argparse | Defines command-line options for preparation, training, fine-tuning, and evaluation. |
| Bash | Provides workflow wrappers under `script/`, using an explicit virtual-environment interpreter and portable paths. |
| Slurm and PyTorch `torchrun` | The inherited multi-GPU wrapper requests cluster resources through Slurm and launches distributed PyTorch workers. It contains machine-specific settings that need adaptation before use. |
| Git | Tracks source code and configuration changes. |

Sources: [configuration utilities](../src/utils/config_utils.py), [YAML configurations](../src/configs/multi_modal/), and [multi-GPU launcher](../script/train_multi_gpu.sh).

## Dependency scope

[requirements.txt](../requirements.txt) covers the current spike/vision application.
[requirements-lfp.txt](../requirements-lfp.txt) adds the inherited optional raw LFP
path. `timm` and `torchvision` are not needed by the current source or CLIP
extractor. Seaborn remains an indirect dependency of ibllib.

Platform constraints record resolved versions, including optional/transitive
packages; a constraint does not cause a package to be installed. See
[environment setup](environment.md) and [validation results](environment-validation.md)
for compatibility pins and outstanding checks.
