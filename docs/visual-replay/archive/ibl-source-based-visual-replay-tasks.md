# Source-based, mouse-perspective replay: atomic tasks

Updated: **2026-09-22**. Status: **VR01–VR06 complete; VR09 selected; VR08 not applicable under the current decision; other tasks pending**.

This checklist implements the [source-based replay specification](ibl-source-based-visual-replay-spec.md) and the agreed [task plan](visual-renderer-tasks.md). It covers stimulus generation, a schematic mouse viewpoint, and matching CLIP inputs. Creating this document does not complete any renderer task.

## Direction and completion rules

- Attempt execution and capture of the original IBL `Gabor2D.bonsai` workflow first. Use a source-derived Python implementation only when the documented feasibility decision requires it.
- Render a minimal schematic scene from a fixed camera at the nominal midpoint between the mouse's eyes, looking at a physical screen with neutral surroundings. This is an approximate viewpoint, without eye tracking, retinal optics, or reconstructed ambient illumination.
- Use scene frames for CLIP and retain stimulus-only reference captures separately. Keep source-display mapping, physical screen placement, and camera projection as distinct stages.
- Follow specification section 18: session evidence, relevant historical source, task configuration, publication, manufacturer specification, retrospective measurements, then project assumptions. Every fallback needs provenance.
- Use manual verification, source inspection, and existing local checks. Do not write or modify tests, temporary test scripts, CI/CD configuration, or automation for those systems. Keep generated videos, captures, datasets, and model artifacts outside source changes.
- Mark a task complete only when its outcome and acceptance evidence are recorded. Record blockers explicitly. An inactive conditional task is **not applicable**, with the VR02 decision linked; it is not an implemented task.

VR08 and VR09 are alternative production paths selected by VR02. There is no automatic runtime switch from a failed historical renderer to Python. Reference captures from the original workflow remain necessary to validate a Python port, even when that workflow is unsuitable for production. If no captures can be obtained, VR10 remains blocked and downstream work cannot be declared fidelity-validated.

## Current integration points

The current [generator](../src/visual_stim_gen.py) uses a 720 x 720 canvas, 30 FPS, linear angular mapping, and an approximate pixel sigma. It already preserves original trial IDs, contrast, wheel direction, freeze events, timestamp sidecars, and staged session publication. Preserve these safeguards while replacing rendering assumptions.

The [parameter loader](../src/utils/stimulus_parameters.py) accepts normalized manifests; raw parameter recovery needs verified units and clock mapping. The [extractor](../src/prepare_visual_stim.py) directly imports the current rendering functions for its render mode and assumes regular video timing. Both interfaces must be addressed. Existing usage is described in [visual extraction](visual-extraction.md).

## Dependency overview

| Task | Outcome | Depends on |
| --- | --- | --- |
| VR01 | Pinned upstream source inventory | None |
| VR02 | Original-renderer feasibility decision | VR01 |
| VR03 | Rendering and control semantics | VR01 |
| VR04 | Session evidence inventory | None |
| VR05 | Normalized parameter recovery | VR03, VR04 |
| VR06 | Backend-neutral replay contract | VR03, VR05 |
| VR07 | Source-frame timing and stimulus state | VR06 |
| VR08 | Historical workflow capture, conditional | VR02, VR07 |
| VR09 | Python production port, conditional | VR02, VR03, VR07 |
| VR10 | Stimulus-renderer fidelity evidence | Selected VR08 or VR09 path |
| VR11 | Schematic viewpoint profile | VR03 |
| VR12 | Mouse-perspective scene renderer | VR10, VR11 |
| VR13 | Video and provenance publication | VR06, VR12 |
| VR14 | Scene-based CLIP extraction | VR13 |
| VR15 | Representative-session validation | VR14 |
| VR16 | Regeneration and migration guide | VR15 |

## Atomic implementation tasks

### VR01 — Audit and pin upstream rendering sources

**Status:** Complete (2026-09-19; source audit only). **Dependencies:** None. **Subsystem:** Upstream source provenance and dependencies.

**Evidence:** [Source audit and acquisition guide](ibl-renderer-source-audit.md), [machine-readable source lock](ibl-renderer-sources.lock.json). Pinned three repositories, inventoried 93 files and 56 configured packages, and hashed the portable Bonsai and BonVision downloads. Runtime compatibility and session revision matching remain unverified for VR02/VR04.

**Outcome:** A reproducible source inventory for the original renderer.

**Acceptance criteria:**

- Resolve the specification's historical IBLRIG branch to an immutable commit; inventory `Gabor2D.bonsai`, its layout, OSC configuration, package configuration, included BonVision workflows, and shaders.
- Record the compatible Bonsai/BonVision versions, source or package hashes, acquisition instructions, licenses, and attribution requirements. Treat the specification's version numbers as evidence to verify.
- Distinguish the chosen reference revision from the revision used by each recording session; record unknown applicability and reconcile it with VR04 when evidence arrives.
- Review any dependency additions against the repository's platform constraints; keep the historical rendering environment isolated from the Python application environment.

### VR02 — Establish original-renderer feasibility

**Status:** Complete (2026-09-19; failed-launch assessment and explicit backend decision). **Dependencies:** VR01. **Subsystem:** Historical runtime and capture feasibility.

**Evidence:** [Feasibility decision and reproduction guide](ibl-renderer-feasibility.md), [runtime/package evidence](ibl-renderer-feasibility-evidence.json). Restored 55 retained configured packages after excluding unavailable PointGrey; original workflow execution failed opening absent COM7, followed by an OpenTK disposal exception. No controlled frames were captured. VR09 is selected for this environment; original-reference capture remains unverified and required for VR10.

**Outcome:** An evidence-backed decision selecting original capture or the Python production path.

**Acceptance criteria:**

- Attempt a minimal launch with the pinned environment, then controlled blank and visible-stimulus inputs, without physical wheel/task hardware.
- Record input delivery, required displays/GPU/runtime, capture access, observed dimensions, frame completeness, and whether capture can associate frames with requested replay states.
- Document every adaptation, including window/display changes and disabled hardware outputs; do not assume a desktop screenshot is a faithful framebuffer capture.
- Select VR08 if repeatable input and complete capture are viable. Otherwise record the concrete blocker and select VR09. Record reference-only capture feasibility separately from production feasibility.
- Finish with a decision record and reproducible attempt instructions, even if the environment cannot run. Do not silently change backends during later generation.

### VR03 — Trace rendering and control semantics

**Status:** Complete (2026-09-20; static source trace). **Dependencies:** VR01. **Subsystem:** Display mapping and task control.

**Evidence:** [Rendering and control semantics](ibl-renderer-semantics.md). Traces the projection/framebuffer chain, OSC and runtime parameter conversions, sigma/blending, wheel reset/gain, event-specific freeze behavior, and synchronization square. Includes derived -35/0/+35-degree positions and source-specific phase/angle discrepancies. All 93 locked source blobs verified; original-frame validation remains VR10 work.

**Outcome:** A source-linked description of the complete parameter-to-frame transformation.

**Acceptance criteria:**

- Trace OrthographicView, SphereMapping, ViewWindow, NormalizedView, DrawGratings, and relevant shader stages; identify the final display framebuffer to capture.
- Establish coordinate signs, angle/phase units, carrier orientation, contrast scaling, Gaussian sigma/aperture interpretation, clipping, and clear-color behavior from the pinned source.
- Trace OSC/task parameter inputs, gain units, wheel reference and coupling interval, freeze/response behavior, and offset. Distinguish controller logic from renderer inputs so wheel motion is applied once.
- Document how -35 degrees, 0 degrees, and +35 degrees should map through this chain. Do not interpret BonVision scene dimensions as centimetres without evidence.

### VR04 — Inventory session-specific evidence

**Status:** Complete (2026-09-22; local files and cached metadata inventoried; live remote sources uninspected). **Dependencies:** None. **Subsystem:** Session data discovery.

**Evidence:** [Per-session and per-dataset inventory](ibl-session-evidence.json), produced by [the read-only inventory command](../src/inventory_visual_evidence.py). No renderer or parameter-recovery changes are included in VR04.

**Outcome:** An availability table for the configured EIDs.

**Acceptance criteria:**

- Inspect the repository's configured session lists and available metadata for raw task/visual parameters, task software versions, display calibration, wheel samples, and synchronized display events.
- For each dataset, record EID, collection/revision where available, source location, clock/units evidence, and availability as available, unavailable, or uninspected.
- Cover the specification's raw fields, including trial number, initial position, contrast, frequency, angle, gain, sigma, phase, and Bonsai timestamps.
- Identify a session suitable for VR15 and its known gaps. Do not report a failed lookup or uninspected source as proof that data never existed.

#### VR04 observations and availability

Inspected all **20 unique EIDs**: 18 train and 2 test, with no overlap. Each has a cached Alyx session response and a readable local ALF trial table. The JSON records list membership, metadata source and hash, session path, task protocol, dataset URLs/IDs, collections, catalog revisions, actual local revision directories, file hashes, array dimensions, finite-value counts, and table fields. Paths are relative to the recorded cache root. No research payloads were changed or downloaded.

Availability is scoped: **available** means a local file or metadata record was found; its separate content status says whether it was read. **Unavailable** means absent from the inspected local session directory, not nonexistent historically. **Uninspected** covers current remote access, remote payloads, and unresolved evidence. Cached catalog entries are discovery leads, not successful downloads. Their cache expiry is not their acquisition date. A filename-based inventory cannot rule out differently named or privately held evidence.

All local trial tables are in collection `alf`. `dated` below means local revision directory `#2025-03-03#`; `base` means no revision directory. Catalog revision identifiers are retained separately and must not be assumed equivalent to local date labels. All rows have local stimulus offset arrays. Wheel means both local position and timestamp arrays.

| EID | Subject | Task version in metadata | Trial rows | Table revision | Local wheel |
| --- | --- | --- | ---: | --- | --- |
| 754b74d5-7a06-4004-ae0c-72a10b6ed2e6 | NYU-29 | 6.4.2 | 1028 | base | available |
| 781b35fd-e1f0-4d14-b2bb-95b7263082bb | KS044 | 6.4.2 | 471 | dated | unavailable |
| 4b00df29-3769-43be-bb40-128b1cba6d35 | CSHL052 | 6.2.5 | 812 | dated | unavailable |
| 7cb81727-2097-4b52-b480-c89867b5b34c | SWC_052 | 6.4.2 | 658 | dated | unavailable |
| 0c828385-6dd6-4842-a702-c5075f5f5e81 | UCLA015 | 6.4.2 | 610 | dated | available |
| b196a2ad-511b-4e90-ac99-b5a29ad25c22 | KS084 | unspecified | 530 | base | unavailable |
| aad23144-0e52-4eac-80c5-c4ee2decb198 | KS023 | 6.1.3 | 641 | dated | unavailable |
| 61e11a11-ab65-48fb-ae08-3cb80662e5d6 | NYU-21 | 6.4.1 | 891 | base | unavailable |
| 7af49c00-63dd-4fed-b2e0-1b3bd945b20b | NYU-37 | 6.4.2 | 414 | dated | unavailable |
| b03fbc44-3d8e-4a6c-8a50-5ea3498568e0 | DY_010 | 6.2.5 | 402 | base | unavailable |
| d0ea3148-948d-4817-94f8-dcaf2342bbbe | ZFM-01936 | 6.4.2 | 560 | dated | unavailable |
| 8928f98a-b411-497e-aa4b-aa752434686d | KS096 | 6.5.3 | 663 | dated | unavailable |
| ecb5520d-1358-434c-95ec-93687ecd1396 | CSHL051 | 6.2.5 | 510 | dated | unavailable |
| 746d1902-fa59-4cab-b0aa-013be36060d5 | ZFM-01592 | 6.4.2 | 660 | dated | unavailable |
| 91bac580-76ed-41ab-ac07-89051f8d7f6e | UCLA044 | 6.4.2 | 699 | dated | unavailable |
| 0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63 | KS045 | 6.4.2 | 314 | base | unavailable |
| 6899a67d-2e53-4215-a52a-c7021b5da5d4 | MFD_06 | unspecified | 402 | base | unavailable |
| 6f09ba7e-e3ce-44b0-932b-c003fb44fb89 | SWC_043 | 6.4.2 | 551 | dated | unavailable |
| 54238fd6-d2d0-4408-b1a9-d19d24fd29ce | DY_018 | 6.4.2 | 623 | dated | unavailable |
| 73918ae1-e4fd-4c18-b132-00cb555b1ad2 | ibl_witten_27 | 6.4.2 | 625 | dated | unavailable |

For **every listed EID**, raw task/encoder logs, display calibration and raw synchronization arrays are unavailable locally under the searched names. Raw task/encoder and calibration files were also absent from the inspected cached catalogs; current remote/private archives remain uninspected. Synchronization and wheel datasets that are cataloged retain their source URLs in the JSON. ALF stimulus events are available, but are not a captured frame sequence or a verified Bonsai-to-ALF clock map.

| Required raw field | Local availability for all 20 EIDs | Evidence and remaining work |
| --- | --- | --- |
| `trial_num` | unavailable | ALF row identity exists; mapping to raw trial numbers is unverified. |
| `stim_pos_init` | unavailable | Left/right contrast fields indicate side, not measured initial angular position. |
| `stim_contrast` | unavailable as raw field | ALF `contrastLeft`/`contrastRight` are available fractional contrasts; retain original row and side. |
| `stim_freq` | unavailable | Do not label the protocol's cycles/degree default as recovered session evidence. |
| `stim_angle` | unavailable | Session value, unit and actual renderer application remain unverified. |
| `stim_gain` | unavailable | Resolve units and sign against the session's historical controller; do not infer from wheel positions. |
| `stim_sigma` | unavailable | Session aperture value and conversion remain unverified. |
| `stim_phase` | unavailable | Any synthetic phase must remain an explicit fallback. |
| `bns_ts` | unavailable | No raw Bonsai timestamps or verified conversion into synchronized ALF time. |

Clock/unit evidence: the installed `ibllib/io/extractors/ephys_fpga.py` documents wheel timestamps in seconds and position in radians, and conversion from Bpod into FPGA time. These are format/extractor conventions; this inventory does not reproduce each session's extraction. The [IBL raw-loader documentation](https://docs.internationalbrainlab.org/_autosummary/ibllib.io.raw_data_loaders.html) describes raw encoder ticks, encoder timestamps, Bonsai datetimes and the nine trial-info fields. Its gain description must be reconciled with the executable historical source, as noted in VR03. Do not mix raw clocks with ALF seconds or treat the stimulus-event times as all display refreshes.

The metadata reports `_iblrig_tasks_ephysChoiceWorld` with the suffixes above; two records omit a version. **None establishes the exact Bonsai graph, package versions or commit used by that session.** The selected v8 source reference is therefore not session-verified. VR05 must reconcile the historical 6.x controllers and renderer behavior before treating v8 defaults/conversions as applicable.

#### VR15 candidate and handoff

Select **UCLA015, `0c828385-6dd6-4842-a702-c5075f5f5e81`** (train session), provisionally for end-to-end work. Its local table and offsets are in `churchlandlab_ucla/Subjects/UCLA015/2022-03-31/001/alf/#2025-03-03#/`; its wheel arrays are in the parent `alf/` collection. All 610 finite onset-to-offset intervals fall within the wheel range. Both wheel arrays contain 758,935 finite samples; timestamps have no duplicates or backward steps. Both sides, correct/error outcomes and contrasts 0, 0.0625, 0.125, 0.25 and 1 are represented. Contrast 0.5 is absent and must be covered by VR10 controlled inputs.

Starting original ALF row IDs: left 0/2/4, right 1/3/5, zero contrast 4/5/11, correct 0/1/2, error 4/11/12. These are inspection candidates, not validated rendered trials. Outward motion, freeze behavior and frame timing still require inspection. Phase, gain, initial position, calibration, clock mapping and exact workflow revision remain unresolved; original reference frames are still required by VR10. Candidate suitability does not mean VR15 is ready to pass.

NYU-29 is an alternate with 2,201,621 finite wheel samples, 484 duplicate timestamp steps, no backward steps, and 1,027/1,028 onset-offset intervals covered. Original row 1027 has missing timing. Preserve the existing duplicate handling and original trial IDs; do not silently remove/renumber trials.

#### Reproduction and verification

From the checkout, using an environment with NumPy, pandas and a pandas Parquet engine:

```powershell
python src/inventory_visual_evidence.py --output tmp/vr04-session-evidence.json
```

The command honors `VINED_DATA_DIR`, supports `--cache-dir`, makes no network requests, and refuses to overwrite an existing output; choose a fresh output name for another run. It reads relevant local files and cached session responses, so results depend on cache contents. Source/list/file hashes make the observations traceable. It does not fetch missing files, recover raw parameters, or claim remote absence.

Verification performed: actual inventory run across 20 sessions, manual array/table inspection for both candidates, and source/JSON/diff checks. No test code added. The existing `.venv` launcher failed because it points to an unavailable Windows Store Python; inventory ran with the installed Python 3.12 and its NumPy/pandas/Parquet environment. No project dependencies were changed. Context7 returned unrelated libraries for ONE-api, so local IBL source and official raw-loader documentation supplied the format evidence.

### VR05 — Extend normalized parameter recovery

**Status:** Complete (2026-09-22; recovery interface implemented and available ALF evidence exercised; real raw-log recovery remains unverified because local raw logs are unavailable). **Dependencies:** VR03, VR04. **Subsystem:** Parameter loading and normalization.

**Evidence:** [Recovery command](../src/recover_stimulus_parameters.py), [normalized loader](../src/utils/stimulus_parameters.py), [replay integration](../src/visual_stim_gen.py), and [input contract/migration guidance](visual-extraction.md#parameter-recovery-vr05).

**Outcome:** Recoverable session/trial evidence feeds the normalized parameter interface.

**Acceptance criteria:**

- Map raw trial identifiers to original ALF rows explicitly; preserve IDs across missing/invalid trials and verify raw phase, orientation, gain, and distance units before conversion.
- Synchronize raw timestamps before using them with ALF times. Unsupported units, mappings, or clock conversions fail explicitly rather than being inferred silently.
- Apply the source hierarchy per parameter. Use source-supported defaults of 0.1 cycles/degree, 7-degree sigma, zero-degree orientation, and the standard position/gain only where applicable; retain observed contrast and phase when recoverable.
- Preserve deterministic synthetic phase only as a labeled fallback, with seed and identity inputs recorded. Preserve strict parameter mode and document any manifest migration.
- Record effective values, original evidence, conversions, and fallback reasons at the appropriate session/trial level; do not classify all omitted values as the same kind of approximation.

**Implemented:** Schema-v2 field-level evidence follows the specification's source hierarchy; v1 remains readable as a legacy declaration. Raw encoderTrialInfo SSV/CSV import requires declared units and unit evidence, an explicit one-to-one raw-to-ALF row map, reasoned exclusions, and measured clock anchors. It refuses unsupported units, conflicting evidence, duplicate mappings, backward clocks and extrapolation. Missing/invalid ALF rows retain their IDs. Observed contrast is cross-checked, recovered phase is preserved, signed per-trial gain is applied once, and task inputs remain distinct from legacy shader interpretation. Recovered row IDs are bound to the ordered ALF table contents; a different table is rejected before generation.

The pinned reference profile supplies 0.1 cycles/degree, task sigma 7 degrees, angle zero and standard signed position/gain only after explicit session-applicability and reversal declarations. No configured 6.x session was automatically assigned that v8 profile. Missing raw phase uses the existing deterministic phase draw, now recording seed, EID, original row, hash and NumPy version. Radius, gain, FOV, position, frequency, angle and legacy pixel sigma each receive their own fallback explanation. Strict v2 mode rejects missing values and project/synthetic fallbacks.

**Verification:** Recovered and loaded 610 original UCLA015 rows (0–609) from the explicitly selected `alf/#2025-03-03#/_ibl_trials.table.pqt`; observed contrasts were recovered and no raw values invented. Strict recovery rejected this incomplete evidence at row 0 before creating its output. Manually generated sidecars for original rows 0, 1, 4, 11 and 12, with 41, 45, 229, 80 and 102 frames respectively. Row 12's MP4 passed the existing complete-decode/frame-count check. Metadata distinguishes session contrast, synthetic phase and individual approximations. CLI help and syntax checks passed; no tests or dependencies were added/modified.

Runtime: project Python 3.10.11, NumPy 1.26.4, pandas 2.3.3, OpenCV 4.10.0. Sandbox access initially prevented runtime/OpenCV inspection; verification succeeded with approved access outside the sandbox. The final recovered manifest is ignored at `tmp/vr05-ucla015-final.json`, SHA-256 `83ef0a025c3d418e6f4c12a4d4482dd9122ab63ea880d91051e55f141128dec4`; replay observations remain at `tmp/vr05-project-python/observations.json`. The table content fingerprint is `214cdc6ea57f9b5a7bf91d5b99dac4600c3c4837beec43dd9f844803e303d5e4`. These artifacts demonstrate the available-data path, not original-renderer fidelity. Raw SSV/CSV, source-profile selection and clock conversion were inspected in code but could not be exercised against a recovered session log. Context7 had no ibllib entry; the installed IBL loader, pinned VR03 source and official raw-loader documentation supplied the format evidence.

**Remaining session gaps:** raw task parameters, phase, encoder reset/coupling events, processed-wheel sign mapping, calibration and actual renderer revision remain unresolved as recorded by VR04. The legacy trajectory and shader interpretation stay explicitly approximate. Parameter-log times are not display-frame timestamps; VR06–VR10 still own the replay state/timing/backend and original-frame comparison work.

### VR06 — Define backend-neutral replay records

**Status:** Complete (2026-09-22; contract and legacy adapter integrated; source timing/backend/scene work remains in later tasks). **Dependencies:** VR03, VR05. **Subsystem:** Replay interfaces and artifact schema.

**Evidence:** [Shared contract and validators](../src/utils/replay_contract.py), [producer integration](../src/visual_stim_gen.py), [consumer integration](../src/prepare_visual_stim.py), and [contract/compatibility reference](visual-extraction.md#replay-contract-vr06).

**Outcome:** A versioned contract shared by timing, rendering, publication, and extraction.

**Acceptance criteria:**

- Define resolved trial parameters, frame stimulus state, source-frame index and session timestamp, validity/reason, backend identity, source hashes, and a versioned scene-profile reference.
- Distinguish recorded display times, reconstructed source times, and encoded-video presentation times. Define the mapping when irregular source times become constant-FPS video.
- Specify separate stimulus-frame and scene-frame outputs and backend capabilities, including capture storage and direct-frame support.
- Record compatibility and rejection behavior for existing parameter manifests, replay metadata, and NPZ sidecars; preserve original IDs and session-second precision.
- Fingerprint all rendering dependencies and the scene profile, replacing reliance on a hash of only the generator entry point where necessary. VR11 supplies the profile's concrete geometry before scene output is produced.

**Implemented:** Replay contract schema 1 binds backend identity/capabilities, dependency hashes, source clock classification, video mapping and a versioned scene-profile reference. Trial records bind original IDs, resolved parameters, provenance, validity/reason and source frame counts. NPZ sidecars preserve float64 source session times, int64 source indices, stimulus state/position/contrast/phase, frame validity/reasons, and separate float64 video PTS with an int64 video-to-source mapping. Contract and record fingerprints link metadata to sidecars. The mapping contract defines both identity and hold-previous semantics; the current producer/extractor supports identity only and rejects other backend/scene/mapping requests without fallback.

The active backend remains `legacy_python_gabor` version 1, with explicit approximate state codes and reconstructed 30 Hz source times. No source state machine or source-faithful renderer is claimed. Stimulus and scene outputs are distinct artifact spaces; scene geometry and original lossless capture storage remain null. VR11 must supply concrete geometry before scene output is possible. Capture format/index requirements are specified for later capture adapters.

Dependency identity now covers seven local rendering/adapter/helper files, Python/platform identity, NumPy/OpenCV versions and installed package artifact-manifest hashes, plus OpenCV build information. Direct rendering checks this dependency set rather than only `visual_stim_gen.py`. Encoded-video reading preserves its recorded dependency provenance. Parameter manifests v1/v2 and feature archives v2 retain their existing roles; unversioned replay manifests/sidecars require regeneration into a fresh root. No old artifacts were rewritten.

**Verification:** Generated original UCLA015 rows 0 and 1 into ignored `tmp/vr06-final/`. Row 0 has 41 source frames and 41 encoded frames; row 1 has 45 source frames and no video mapping. The existing complete-decode check passed for row 0's MP4. Video and direct frame consumers each sampled seven frames for row 0; direct mode sampled eight for row 1. Loaded times remained float64 and legacy state codes were 240/241. A real earlier dependency snapshot was rejected by the direct-render check after the adapter changed. Syntax, JSON/NPZ loading and whitespace checks passed. Contract fingerprint: `d5773af9c22ad22b60d419271fe55d96f3e0a25b0f032ebf01973219c85097e4`. Verification used the existing project Python outside the sandbox with approved access; no test code, dependency changes or CLIP model run were added. Context7's NumPy documentation confirmed NPZ dtype preservation and pickle-free numeric/Unicode loading.

**Handoff:** VR07 implements source timing/state semantics; VR08/VR09 implement the selected backend; VR11 supplies scene geometry; VR13/VR14 implement nonidentity publication/extraction and scene inputs. The generic validators do not certify source fidelity, captured-frame completeness or scene geometry, and no such outputs were produced here.

### VR07 — Implement source-frame timing and stimulus state

**Status:** Pending. **Dependencies:** VR06. **Subsystem:** Replay timeline and trajectory.

**Outcome:** A backend-independent sequence of timestamped stimulus states.

**Acceptance criteria:**

- Use recorded display timing where it resolves the required events/frames. Do not infer a complete measured frame clock from onset/offset markers alone; label any 60 Hz reconstruction explicitly.
- Preserve source timing evidence and implement the verified wheel coupling, direction, inward/outward trajectories, freeze, and visibility interval `[stimOn, stimOff)`.
- Retain wheel coverage checks, duplicate-sample handling, invalid trial reasons, and original IDs. Do not extrapolate missing observations into apparently measured states.
- Represent known blank-display states correctly if requested; do not mark unknown or uncovered intervals as valid gray observations. Keep existing neural alignment semantics intact.

### VR08 — Implement historical workflow capture

**Status:** Not applicable under the [current VR02 decision](ibl-renderer-feasibility.md); reconsider only with demonstrated hardware-free capture. **Dependencies:** VR02, VR07. **Subsystem:** Historical renderer adapter.

**Outcome:** Lossless stimulus frames captured from the original rendering path.

**Acceptance criteria:**

- Supply resolved parameters and replay state through the verified task/OSC interfaces, retaining the original geometry and shaders.
- Associate each capture with its intended source state and session time; keep host capture time distinct. Detect dropped, repeated, or reordered captures through frame/state identity rather than pixel differences alone.
- Capture the final stimulus framebuffer without desktop overlays, cursor, or unrelated windows. Record actual dimensions, pixel format, capture stage, and deviations from the historical environment.
- Fail incomplete capture explicitly and preserve recoverable artifacts. Confirm repeated identical inputs produce consistent captures before publishing a replay.

### VR09 — Implement the conditional Python port

**Status:** Pending; selected by the [VR02 decision](ibl-renderer-feasibility.md). **Dependencies:** VR02, VR03, VR07. **Subsystem:** Python rendering backend.

**Outcome:** A source-derived implementation of the original display rendering path.

**Acceptance criteria:**

- Reproduce the traced projection, shader/carrier, contrast, phase, Gaussian/aperture, clipping, and RGB(128,128,128) clear behavior; replace the arbitrary square canvas and independent pixel sigma.
- Preserve a source-compatible 4:3 render target, source timing, and angular parameter meanings. Document every implementation deviation with the relevant source reference.
- Implement the VR06 interface without replacing source geometry with an independent linear degrees-to-pixels approximation.
- Record fidelity as unverified until VR10 succeeds against original captures. Arrange reference-only capture separately if the original runtime cannot support production; unavailable reference images are a blocker to fidelity certification.

### VR10 — Establish stimulus-renderer fidelity

**Status:** Pending. **Dependencies:** Selected VR08 or VR09 path. **Subsystem:** Renderer validation evidence.

**Outcome:** A recorded comparison of stimulus-only outputs against the historical workflow.

**Acceptance criteria:**

- Obtain original-workflow lossless reference captures with pinned versions and known inputs. For VR08, compare adapter-controlled output with independently controlled reference captures; for VR09, compare Python output with those references.
- Cover every stimulus case in the verification matrix below. Record image differences and geometric measurements, including background, carrier period, envelope, center position, clipping, and cadence.
- State comparison alignment, metrics, and justified tolerances before assessing candidate results; retain maximum/mean pixel differences and geometric errors rather than only a pass label.
- Record discrepancies and their causes. Do not use lossy MP4 differences to certify shader fidelity, or describe a matching renderer as proof of exact session reconstruction.
- Complete only when the acceptance cases meet the documented criteria. If references are unavailable or discrepancies remain material, record a blocker; retain the unverified label.

### VR11 — Define the schematic viewpoint profile

**Status:** Pending. **Dependencies:** VR03. **Subsystem:** Physical scene and virtual camera configuration.

**Outcome:** A concrete, versioned scene profile ready for implementation.

**Acceptance criteria:**

- Specify camera origin at the nominal eye midpoint, fixed pose toward the screen, coordinate axes/units, projection, horizontal/vertical field of view, output dimensions, and screen plane position/orientation/dimensions.
- Use a minimal screen and neutral unlit surround; record its digital appearance as a project visualization choice. Omit unsupported rig details, retinal optics, and eye/head motion.
- Reconcile the specification's panel dimensions, pixel pitch, approximate 8 cm distance, and approximate 102-degree coverage. Derive coverage from the chosen consistent geometry and document discrepancies rather than forcing incompatible values to agree.
- Distinguish workflow render dimensions, physical panel dimensions/native resolution, and final camera output. Include the complete screen plus surroundings in the camera framing.
- Record exact chosen values and evidence/fallback status; no unspecified camera or geometry settings remain at completion. Do not interpret BonVision's 20/15/-7 scene values as physical measurements without verification.

### VR12 — Render the mouse-perspective scene

**Status:** Pending. **Dependencies:** VR10, VR11. **Subsystem:** Scene projection.

**Outcome:** Timestamp-preserving scene frames viewed from the nominal mouse position.

**Acceptance criteria:**

- Place the final source display image on the physical screen plane, then apply the fixed camera projection; do not reapply spherical display correction to an already mapped framebuffer.
- Keep the display unlit/emissive in the schematic rendering sense, without invented gamma, luminance, shadows, or ambient-light simulation. Document texture filtering/resampling.
- Verify screen corners, center, aspect ratio, and left/right stimulus placement against the profile; preserve source-frame identity and timestamps.
- Retain separate stimulus-only outputs and scene outputs. Include the scene-profile identity in each generation's provenance and label the scene approximate even when the stimulus backend is validated.

### VR13 — Publish videos and provenance

**Status:** Pending. **Dependencies:** VR06, VR12. **Subsystem:** Encoding, sidecars, and session publication.

**Outcome:** Complete perspective videos with traceable frame mappings and metadata.

**Acceptance criteria:**

- Preserve `trial_<original row>.mp4` naming, timestamp sidecars, invalid-trial records, fresh-output protection, staging, and publication only after validation; retain unfinished output on failure.
- Use 60 FPS as the default storage cadence for reconstructed 60 Hz input. Record any other storage cadence, irregular-to-regular conversion, repeated/dropped source frames, and mapping to source session times explicitly.
- Decode finished videos to verify frame counts, dimensions, cadence, and sidecar consistency. Store lossless stimulus references separately from inspection MP4s.
- Record all applicable specification-section-16 metadata plus backend/source versions, parameter provenance, scene profile, actual render dimensions, output dimensions, codec, and resampling.
- Keep physical luminance null unless measured; classify 29.4 cd/m² only as a retrospective average. Record unresolved fullscreen scaling, session gamma, brightness, ambient illumination, and calibration without inventing values.

### VR14 — Integrate perspective frames with CLIP

**Status:** Pending. **Dependencies:** VR13. **Subsystem:** Frame sources and feature extraction.

**Outcome:** CLIP features represent the same perspective scene as the videos.

**Acceptance criteria:**

- Replace direct-render coupling to the old patch functions with the versioned frame/backend contract. Video and supported direct-frame inputs use the same scene, parameters, identities, and source-time mapping.
- Explicitly reject unsupported combinations such as direct regeneration from a capture-only backend without required captured frames. Do not substitute the legacy renderer when captures are missing.
- Fit the complete scene into the encoder input while preserving aspect ratio, using explicit padding and no peripheral center crop. Record resize interpolation, padding color, and the full image processor configuration.
- Preserve pinned CLIP revision, normalized 768-dimensional features, schema-v2 feature payload compatibility, original IDs, session timestamps, validity masks, and bounded streaming memory.
- Inspect the actual preprocessed CLIP inputs for left/right peripheral stimuli. Record feature differences between decoded-video and lossless inputs separately from geometry/timing correctness; they need not be numerically identical.

### VR15 — Validate a representative session

**Status:** Pending. **Dependencies:** VR14. **Subsystem:** End-to-end manual validation.

**Outcome:** A reproducible validation report for one selected real session.

**Acceptance criteria:**

- Record EID, dataset revisions, commands, environment, backend, source versions, scene profile, extraction settings, and selected original trial IDs.
- Inspect available left/right, low/zero/high contrast, correct/error, outward-motion, freeze, and missing-data cases. State absent session cases and cover the corresponding rendering behavior with VR10's controlled inputs.
- Verify stimulus capture, scene geometry, MP4 decoding, frame-to-session-time mappings, CLIP preprocessing, and downstream IDs/masks independently; confirm invalid or missing trials do not shift later IDs.
- Record numerical image/geometric comparisons, frame counts, timing discrepancies, and feature-mode differences with limitations. Run applicable existing local checks without adding test code.
- Confirm a new feature generation can pass through alignment and cache preparation without changing the intended neural time grid. Preserve separate artifact locations and report unresolved blockers rather than claiming completion.

### VR16 — Document regeneration and migration

**Status:** Pending. **Dependencies:** VR15. **Subsystem:** User documentation and artifact lifecycle.

**Outcome:** A verified runbook for reproducing the new visual representation.

**Acceptance criteria:**

- Document commands for the selected backend, parameter recovery, scene configuration, replay generation, CLIP extraction, alignment, and cache regeneration using fresh output roots.
- State supported platforms and backend/mode combinations, historical-runtime setup, schema compatibility, and rejection behavior for old artifacts.
- Explain which changes require rebuilding replays, features, aligned data, and caches; retain old generations and avoid combining representations within an experiment.
- Update active visual documentation to distinguish legacy approximations, validated source rendering, and schematic scene assumptions; link validation evidence and remaining unknowns.
- Record actual successful commands from VR15. Do not present proposed or unexecuted commands as validated, and do not add CI/CD instructions.

## Manual verification matrix

| Layer | Required cases | Evidence to retain |
| --- | --- | --- |
| Stimulus pixels | Blank gray; contrasts 0%, 6.25%, 12.5%, 25%, 50%, 100%; several known phases | Lossless original/candidate frames, pixel errors, background and contrast measurements |
| Stimulus geometry | -35, 0, +35 degrees; 0.1 cycles/degree; 7-degree sigma; source orientation; edge clipping; 4:3 aspect | Center/period/envelope measurements in the verified mapping and documented tolerances |
| Temporal behavior | Inward motion, outward/error motion, freeze, onset/offset boundaries, 60 Hz reconstruction, recorded timing where available | Frame/state identities, source timestamps, synchronization evidence, drop/repeat accounting |
| Scene geometry | Screen corners/center, physical aspect and coverage, peripheral stimulus visibility | Profile values, projected coordinates, scene images, evidence that mapping is applied once |
| Encoded video | Complete decode, dimensions, cadence, timestamp correspondence | Counts, source-to-video mapping, codec/resampling record; compression differences assessed separately |
| CLIP input and features | Full-scene preservation, both sides, supported input modes, skipped trials | Preprocessed images, 768-wide features, IDs/times/masks, preprocessing provenance and mode differences |
| Failure handling | Missing parameters/phase/timing, uncovered wheel interval, unsupported backend mode, incomplete capture, existing destination | Explicit failure/fallback records, preserved artifacts, no silent backend replacement or publication of incomplete sessions |

The original-workflow comparison validates stimulus rendering only. The viewpoint remains schematic, and exact original sensory input cannot be claimed without session-specific display, timing, and eye-position evidence.

## Specification coverage

| Specification sections | Tasks |
| --- | --- |
| 1–2, 12, 18–19: confidence, original source, reuse strategy, evidence hierarchy | VR01–VR05, VR08–VR10 |
| 3–6, 13: render window, gray, geometry, grating parameters, obsolete approximations | VR03, VR05, VR08–VR12 |
| 7–10, 15: physical display, luminance, gamma, ambient light, unknowns | VR04, VR11–VR13 |
| 11: source refresh and synchronization | VR06–VR08, VR13–VR15 |
| 14, 16: parameter recovery and metadata | VR04–VR06, VR13–VR14 |
| 17, 20: acceptance cases and implementation decisions | VR10, VR15–VR16 |
| Agreed extension: schematic mouse viewpoint feeds CLIP | VR11–VR16 |
