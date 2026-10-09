"""Fixed-weight CLIP encoding with explicit full-view image preparation."""

from copy import deepcopy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
from dataclasses import dataclass
from pathlib import Path
import math
import hashlib
import re
import time

import numpy as np

from utils.provenance import file_hash, package_versions, source_hashes
from utils.progress import logger
from visual_replay import TrialOutcome
from .observations import ObservationSelection


class FeatureExtractionError(RuntimeError):
    """Encoder/preprocessing failure; no completed feature result is available."""


@dataclass(frozen=True)
class EncodedObservation:
    metadata: dict
    selected: bool
    feature: np.ndarray | None
    status: str


def prepare_image(rgb, size, *, fill=(128, 128, 128)):
    """Fit RGB8 into a square with centered neutral padding, without cropping."""
    from PIL import Image

    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0):
        raise ValueError("Expected a nonempty uint8 RGB image")
    if type(size) is not int or size <= 0:
        raise ValueError("size must be a positive integer")
    height, width = rgb.shape[:2]
    scale = size / max(width, height)
    dimensions = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = Image.fromarray(rgb).resize(dimensions, resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(image, ((size - dimensions[0]) // 2, (size - dimensions[1]) // 2))
    return canvas


class ClipEncoder:
    """Load one immutable model/processor snapshot and encode bounded batches."""

    def __init__(self, model_name="openai/clip-vit-large-patch14", *, revision="main",
                 device=None, workers=1, precision="auto", gpu_duty_cycle=1.0,
                 feature_cache_size=512, attention_backend="auto"):
        if type(workers) is not int or workers < 1:
            raise ValueError("workers must be a positive integer")
        self._workers = workers
        if type(feature_cache_size) is not int or feature_cache_size < 0:
            raise ValueError("feature_cache_size must be a nonnegative integer")
        self._feature_cache_size = feature_cache_size
        self._feature_cache = OrderedDict()
        if precision not in ("auto", "float32", "float16"):
            raise ValueError("precision must be auto, float32, or float16")
        if attention_backend not in ("auto", "eager", "sdpa"):
            raise ValueError("attention_backend must be auto, eager, or sdpa")
        if (isinstance(gpu_duty_cycle, bool) or not isinstance(gpu_duty_cycle, (int, float))
                or not math.isfinite(gpu_duty_cycle) or not 0 < gpu_duty_cycle <= 1):
            raise ValueError("gpu_duty_cycle must be finite and in (0, 1]")
        self._gpu_duty_cycle = float(gpu_duty_cycle)
        import torch
        from huggingface_hub import HfApi, snapshot_download
        from transformers import CLIPModel, CLIPImageProcessor

        logger.info("visual-features: resolving CLIP %s revision %s", model_name, revision)
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            revision = HfApi().model_info(model_name, revision=revision).sha
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("CLIP revision must resolve to an immutable commit")
        logger.info("visual-features: loading/downloading CLIP snapshot %s", revision)
        snapshot = Path(snapshot_download(model_name, revision=revision,
            allow_patterns=["*.json", "pytorch_model.bin", "model.safetensors"]))
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cuda = torch.device(self._device).type == "cuda"
        if self._gpu_duty_cycle < 1 and not cuda:
            raise ValueError("GPU duty-cycle limiting requires a CUDA device")
        if precision == "float16" and not cuda:
            raise ValueError("float16 inference requires a CUDA device")
        self._precision = ("float16" if cuda and torch.cuda.get_device_capability(self._device)[0] >= 7
                           else "float32") if precision == "auto" else precision
        self._attention_backend = ("sdpa" if cuda else "eager") if attention_backend == "auto" else attention_backend
        logger.info("visual-features: loading CLIP weights on %s", self._device)
        self._model = CLIPModel.from_pretrained(str(snapshot), local_files_only=True).to(self._device)
        if self._attention_backend == "sdpa":
            from .attention import use_vision_sdpa
            use_vision_sdpa(self._model)
        self._model.eval()
        self._model.requires_grad_(False)
        self._processor = CLIPImageProcessor.from_pretrained(str(snapshot), local_files_only=True)
        self._size = self._model.config.vision_config.image_size
        self._width = self._model.config.projection_dim
        if type(self._size) is not int or self._size <= 0 or self._width != 768:
            raise ValueError("Require a square-input CLIP model with 768 projected features")
        # Pixel conversion/normalization are explicit; downloaded crop defaults
        # must not undo the full-view adaptation.
        self._processor.do_resize = False
        self._processor.do_center_crop = False
        self._processor.do_rescale = True
        self._processor.rescale_factor = 1 / 255
        self._processor.do_normalize = True
        self._processor.do_convert_rgb = True
        self._provenance = dict(model=model_name, revision=revision,
            artifact_hashes={p.name: file_hash(p) for p in snapshot.iterdir() if p.is_file()},
            output="CLIPModel.get_image_features: projected pooled vision output",
            feature_width=self._width, dtype="float32", normalization="L2 per observation",
            inference=dict(precision=self._precision,
                attention_backend=self._attention_backend,
                policy="CUDA autocast" if self._precision == "float16" else "float32",
                weight_dtype="float32", normalization_dtype="float32",
                image_reuse=dict(capacity=feature_cache_size,
                    policy="SHA256 of prepared RGB8 square; within-batch deduplication and LRU"
                           if feature_cache_size else "disabled")),
            preparation=dict(policy="aspect-preserving fit and centered square padding",
                image_size=self._size, fill_rgb=[128, 128, 128], resample="Pillow BICUBIC",
                rounding="Python round, minimum one pixel", odd_padding="extra pixel at right/bottom",
                input="uint8 RGB [0,255], top row first"),
            processor=self._processor.to_dict(), packages=package_versions(), device=str(self._device),
            sources=source_hashes("src/visual_features/encoder.py", "src/visual_features/observations.py",
                                 "src/visual_features/attention.py"))
        logger.info("visual-features: CLIP ready on %s (%s inference, %d features, %d image workers)",
                    self._device, self._precision, self._width, self._workers)
        logger.info("visual-features: vision attention backend %s", self._attention_backend)
        if self._gpu_duty_cycle < 1:
            logger.info("visual-features: CUDA batch duty cycle %.0f%% (idle pauses enabled)",
                        self._gpu_duty_cycle * 100)

    @property
    def provenance(self):
        return deepcopy(self._provenance)

    def encode(self, frames):
        """Return float32[B,768]; raise rather than publish unusable vectors."""
        import torch

        if not frames:
            raise ValueError("Cannot encode an empty batch")
        try:
            if self._workers > 1 and len(frames) > 1:
                with ThreadPoolExecutor(max_workers=min(self._workers, len(frames))) as pool:
                    images = list(pool.map(prepare_image, frames, [self._size] * len(frames)))
            else:
                images = [prepare_image(rgb, self._size) for rgb in frames]
            keys = []
            resolved = {}
            missing = {}
            if self._feature_cache_size:
                for image in images:
                    key = hashlib.sha256(image.tobytes()).digest()
                    keys.append(key)
                    if key in self._feature_cache:
                        resolved[key] = self._feature_cache[key]
                        self._feature_cache.move_to_end(key)
                    elif key not in resolved:
                        missing.setdefault(key, image)
                if not missing:
                    return np.stack([resolved[key] for key in keys])
                images = list(missing.values())
            inputs = self._processor(images=images, return_tensors="pt")
            if self._gpu_duty_cycle < 1:
                # Wait for earlier work before timing this batch; CUDA launches
                # are asynchronous. The CPU feature copy below waits for completion.
                torch.cuda.synchronize(self._device)
                started = time.perf_counter()
            with torch.inference_mode():
                autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                            if self._precision == "float16" else nullcontext())
                with autocast:
                    features = self._model.get_image_features(
                        **{key: value.to(self._device) for key, value in inputs.items()})
                features = features.float()
                if (features.shape != (len(images), self._width)
                        or not torch.isfinite(features).all()
                        or (torch.linalg.vector_norm(features, dim=-1) <= 1e-12).any()):
                    raise ValueError("CLIP produced invalid projected features")
                features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
            if self._gpu_duty_cycle < 1:
                elapsed = time.perf_counter() - started
                pause = elapsed * (1 / self._gpu_duty_cycle - 1)
                # Short chunks preserve prompt interruption even for a low duty cycle.
                deadline = time.perf_counter() + pause
                while (remaining := deadline - time.perf_counter()) > 0:
                    time.sleep(min(remaining, 1.0))
            if not np.isfinite(features).all():
                raise ValueError("CLIP produced nonfinite normalized features")
            if self._feature_cache_size:
                for key, feature in zip(missing, features):
                    feature = feature.copy()
                    feature.setflags(write=False)
                    resolved[key] = feature
                    self._feature_cache[key] = feature
                    self._feature_cache.move_to_end(key)
                    if len(self._feature_cache) > self._feature_cache_size:
                        self._feature_cache.popitem(last=False)
                return np.stack([resolved[key] for key in keys])
            return features
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            raise FeatureExtractionError(f"CLIP extraction failed: {exc}") from exc


def _observation_batches(observations, batch_size):
    records = []
    for item in observations:
        if isinstance(item, TrialOutcome):
            yield records, item
            records = []
        else:
            records.append(item)
            if len(records) == batch_size:
                yield records, None
                records = []
    if records:
        yield records, None


def _prefetch_batches(batches):
    """One reader thread and at most one batch ahead of the consumer."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="feature-reader") as pool:
        pending = pool.submit(next, batches, None)
        try:
            while True:
                batch = pending.result()
                if batch is None:
                    break
                pending = pool.submit(next, batches, None)
                yield batch
        finally:
            pending.cancel()
        # Executor joins before the caller closes the source iterator.


def iter_encoded_observations(observations, encoder, *, batch_size=32, workers=1):
    """Preserve record order and all source metadata while encoding selected pixels.

    Yield EncodedObservation or unchanged TrialOutcome. Source completion is
    exposed by observations only after exhaustion; this produces no archive.
    """
    if not isinstance(observations, ObservationSelection) or observations.state != "pending":
        raise ValueError("Expected a fresh ObservationSelection")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")

    def encode_batch(records):
        frames = [record.rgb for record in records if record.selected and record.rgb is not None]
        try:
            values = iter(encoder.encode(frames)) if frames else iter(())
        except FeatureExtractionError as exc:
            ids = [record.metadata["observation_id"] for record in records
                   if record.selected and record.rgb is not None]
            raise FeatureExtractionError(f"Observations {ids}: {exc}") from exc
        for record in records:
            feature = None
            if not record.selected:
                status = "not_selected"
            elif record.metadata["status"] != "valid":
                status = "source_unavailable"
            else:
                if record.rgb is None:
                    raise FeatureExtractionError("Selected valid observation has no pixels")
                value = next(values)
                feature = np.frombuffer(value.tobytes(), dtype=np.float32).reshape(value.shape)
                status = "encoded"
            yield EncodedObservation(deepcopy(record.metadata), record.selected, feature, status)

    with observations:
        with closing(_observation_batches(observations, batch_size)) as batches:
            stream = _prefetch_batches(batches) if workers > 1 else batches
            with closing(stream):
                for records, outcome in stream:
                    yield from encode_batch(records)
                    if outcome is not None:
                        yield outcome
