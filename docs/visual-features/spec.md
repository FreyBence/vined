# Visual Features Specification

## 1. Purpose

The visual-features component converts reconstructed visual observations into numerical image representations for modeling the relationship between visual stimuli and neural activity.

It provides reusable feature sequences whose visual source, trial identity, physical timing, and validity remain traceable. Alignment consumes these sequences to combine visual and neural observations.

This document is a proposed specification. It defines required behavior without prescribing package structure, concrete APIs, storage schemas, or implementation algorithms.

## 2. Scope and ownership

Visual-features owns:

- image preparation required by the selected visual encoder;
- feature extraction with a pretrained CLIP image encoder;
- explicit selection of supplied observations when temporal subsampling is requested;
- preservation of observation identity, timing, validity, and source provenance;
- persistence and loading of reusable feature outputs.

Visual-replay owns stimulus reconstruction, scene geometry, mouse-perspective projection, and the meaning of source observation times and validity.

Alignment owns the common visual/neural time grid, temporal interpolation or aggregation, coverage decisions on that grid, and joint modality selection.

Training-dataset and training own model-ready sample construction, padding, experiment splits, training-time masking, and optimization.

## 3. Input requirements

### 3.1 Visual observations

The component must consume observations supplied through the public visual-replay boundary. The intended visual input is the reconstructed mouse-perspective view.

Feature extraction must be possible without producing or decoding an intermediate video. Visual-features must not reconstruct images by accessing renderer internals.

The input must provide enough information to identify:

- the session and original trial;
- the source replay generation and the source trial-table identity;
- the individual source observation;
- its timestamp in the shared session time reference;
- its image format and value range;
- its validity and any available invalidity reason.

The component must also receive the relevant requested domain, coverage information, and known gaps or discontinuities needed to interpret the observations.

Information may be attached to observations or shared at trial or generation level. Concrete field names and representations are defined by the implemented interfaces.

### 3.2 Input interpretation

The component must preserve whether source timing is measured, reconstructed, or assumed. It must not present reconstructed observation times as recorded display refresh times.

Missing information required for correct interpretation must produce an explicit failure or a documented unavailable result. It must not be replaced by silent assumptions about frame rate, trial order, image format, or time origin.

## 4. Feature extraction requirements

### 4.1 Representation

The initial supported feature source is a pretrained CLIP image encoder used with fixed weights during extraction.

The selected representation must be defined by its model, immutable weight identity, extracted model output, dimensionality, and any normalization or other transformation. A dimension such as 768 alone is not a sufficient representation definition.

Each feature must correspond to an identifiable visual observation. Encoder execution must use inference behavior and must not update model weights.

The component must support practical batched processing while preserving observation associations across batch boundaries. Resource management must not require retaining all session images in memory.

Support for additional encoder families, encoder fine-tuning, or a general plugin system is not required.

### 4.2 Image preparation

Image conversion and preprocessing must be explicit and consistent with the selected encoder and representation definition.

The intended spatial extent of the supplied mouse-perspective image must be retained. Preprocessing must not silently crop peripheral stimulus content or distort spatial proportions.

Any resizing, padding, value-range conversion, channel conversion, and normalization must be reproducible from the recorded extraction configuration.

The exact spatial adaptation policy is an architecture decision. Preservation of the supplied field of view does not imply lossless preservation of image detail after resolution reduction.

### 4.3 Observation selection

The component must support extracting features from the full supplied observation sequence. Any temporal subsampling must be explicitly configured and recorded.

Selection must use source time and observation identity consistently across trials and processing batches. Selected observations must retain their actual source timestamps.

The component must not duplicate observations to claim a higher sampling rate or silently reinterpret a requested sampling time as an observed time.

Feature extraction must not resample features onto a neural time grid, interpolate missing embeddings, or assign temporal support beyond what the source provides.

## 5. Identity, timing, and validity

### 5.1 Identity and time

Session identity, original trial identity, and source observation associations must survive extraction, storage, and loading.

Filtering or skipping data must not renumber original trials. Array positions and filenames alone must not determine scientific identity.

Feature timestamps must retain the source session time reference and its precision. If source observations include support intervals, their meaning must remain distinguishable from sample instants.

The component must not infer coverage beyond the last available observation or fill pre-stimulus or post-stimulus periods with fabricated visual data.

### 5.2 Validity and coverage

The output must distinguish:

- valid observations with visible stimulus content;
- valid blank or zero-contrast observations;
- invalid observations;
- unavailable intervals or trials.

Valid blank images must receive their actual visual representation. Missing or invalid data must not be presented as valid blank-image embeddings or valid zero vectors.

Requested trials that contribute no usable features must remain traceable through an explicit status and any available reason.

Temporal subsampling must preserve known invalid regions and continuity boundaries, including those between selected observations. Alignment must be able to identify these limitations without reopening renderer internals.

Known upstream invalidity and feature-extraction failure must remain distinguishable. Nonfinite or otherwise unusable encoder outputs must not be published as valid features.

Validity describes the source and extraction result. Whether a feature can contribute to a particular neural interval is decided by alignment.

## 6. Output, provenance, and reuse

The component must expose loadable feature sequences together with:

- session, original trial, and source observation identities;
- physical timestamps and any source-provided support semantics;
- feature values and their complete representation definition;
- validity, coverage limitations, and trial status;
- source and extraction provenance;
- an identifiable, completed feature generation.

Output must preserve variable sequence lengths without introducing model-specific padding or a fixed neural-bin count. Exact file formats, field names, and reader APIs are implementation contracts rather than requirements of this specification.

Provenance must identify the consumed replay generation and source trial table, encoder weights, effective preprocessing, observation selection, and relevant extraction software/configuration. Source identity must be tied to the actual consumed content rather than only a path or filename.

The output must allow downstream processing to check that visual and neural data refer to compatible session and trial identities.

Existing features may be reused only when compatibility with the requested extraction is established. A change in relevant source content, representation, preprocessing, or sampling must not be silently treated as the same result.

An older generation may remain usable when explicitly selected for reproducing an earlier experiment. Visual-features must expose sufficient identity for downstream datasets to record which generation they consumed; it does not own recursive invalidation of downstream caches.

A result must not appear complete until the requested observations and trial statuses have been accounted for. An interrupted or failed extraction must not expose a partial result as complete or silently return an older artifact as the newly requested result.

## 7. Boundaries and exclusions

This component does not own:

- session acquisition or independent source trial-table loading;
- stimulus reconstruction, renderer geometry, or mouse-camera projection;
- neural preprocessing or multimodal temporal alignment;
- experiment splits, model padding, or training-time masking;
- model training, encoder fine-tuning, or prediction evaluation;
- historical framebuffer certification or video production.

The current renderer, archive schema, 768-dimensional width, sampling cadence, 100-bin window, and 20 ms neural grid are not mandatory compatibility requirements.

Existing implementation may be reused or replaced according to these requirements. Poor historical model metrics do not by themselves establish a feature-extraction defect or determine the required representation.

Scientific data correctness, explicit component boundaries, and reproducibility take priority over speculative infrastructure or optimization.