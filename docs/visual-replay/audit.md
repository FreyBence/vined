# Visual replay audit

## Summary

The implementation provides an explicitly approximate Gabor replay with original trial row IDs, session-clock sidecars, normalized parameter evidence, deterministic synthetic phase, and shared session-data access. It does not yet implement the source-derived display behavior, mouse-perspective observations, or generation/completion boundary required by `spec.md` and `architecture.md`.

This is a source-inspection audit of the active replay generator, parameter recovery/resolution, replay contract, and directly relevant consumer boundary. No session replay or historical-renderer comparison was executed. There is no current `docs/visual-replay/interface.md`; the implemented boundary was inspected in code. Findings describe implementation gaps against the accepted design, not reasons to change that design.

## Findings

### A01 — Historical display behavior is not implemented

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Spec

**Location:**  
`src/visual_stim_gen.py:35–128, 238–249` (`ReplayConfig`, `create_grating_patch`, `render_trial_frame`, backend interpretation); `src/recover_stimulus_parameters.py:162–178` (reference parameter profile).

**Finding:**  
All sessions use a linear degrees-to-pixels mapping, pixel Gaussian envelope, sine phase, and carrier rotation. The metadata explicitly identifies differences from the referenced shader's phase, aperture/blending, and angle interpretation. Parameter recovery can select nominal values from one reference configuration, but this does not select display-rendering behavior. No resolved historical behavior profile controls the renderer. Supplying recovered task parameters therefore still applies legacy semantics to them.

**Expected:**  
Specification “Historical source usage” and “Display reconstruction”; architecture §§3 and 5 require applicable source-derived display semantics resolved before rendering. Exact historical framebuffer equivalence is not required.

**Suggested disposition:**  
Replace the incompatible display mapping with source-derived behavior; retain useful parameter evidence and contrast handling.

---

### A02 — Task state is fixed to onset coupling and a universal freeze fallback

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:**  
`src/visual_stim_gen.py:132–190` (`trial_parameters`, `generate_trial_video`); `src/utils/replay_contract.py:198–199` (legacy states).

**Finding:**  
Wheel displacement always starts at stimulus onset. Missing `stimFreeze_times` always falls back to `response_times`, and the same freeze-in-place rule applies to every session. No profile resolves closed-loop activation, coupling baseline/sign applicability, or version-specific event transitions. State reconstruction is embedded in the fixed 30 Hz video-generation path rather than exposed as a cadence-independent evaluator. Recording these approximations does not establish their applicability to a session.

**Expected:**  
Specification “Position and movement,” “Temporal reconstruction,” and “Freeze behavior”; architecture §4 require one task-state owner using source-appropriate events and movement rules, independently of observation cadence.

**Suggested disposition:**  
Refactor task-state evaluation behind a resolved behavior profile. Preserve the existing exact freeze-time interpolation and outward-motion support.

---

### A03 — Interior wheel gaps silently become valid reconstructed motion

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/visual_stim_gen.py:65–95, 163–164, 188–189`; `src/utils/replay_contract.py:197`.

**Finding:**  
Wheel validation checks ordering and finite values, and interpolation rejects queries outside the first/last samples. It accepts arbitrarily large interior gaps: two samples bracketing a movement interval suffice for a continuous interpolated trajectory. Every emitted frame is then marked valid. There is no explicit admissible-gap policy or unavailable-coverage outcome.

**Expected:**  
Specification “Failure behavior” and architecture §4 require explicit interpolation/gap semantics and prevent unavailable movement evidence from silently yielding valid positions. This does not require treating every naturally sparse wheel interval as invalid.

**Suggested disposition:**  
Refactor coverage handling to use an explicit, evidence-appropriate gap policy and preserve unavailable intervals.

---

### A04 — Mouse-perspective scene projection is missing

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Spec

**Location:**  
`src/visual_stim_gen.py:113–128, 206–214`; `src/utils/replay_contract.py:49–60, 175–184`.

**Finding:**  
The only implemented output is a square stimulus canvas. The contract declares `scene_frames=False`, `scene_profile=None`, and `output_space="stimulus"`; the adapter rejects scene output. There is no physical screen placement, fixed mouse camera, or separately inspectable scene image. The capability declaration is honest, but the intended visual-modality output is absent.

**Expected:**  
Specification “Mouse-perspective scene” and architecture §5 require a distinct scene projection with documented screen geometry/viewpoint and complete relevant screen coverage.

**Suggested disposition:**  
Refactor the rendering pipeline to add the separate scene transformation while retaining display inspection output.

---

### A05 — Direct observations are reconstructed inside the feature consumer

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Architecture

**Location:**  
`src/visual_stim_gen.py:153–272, 429–430`; `src/prepare_visual_stim.py:47–81` (direct consumer boundary); `src/utils/replay_contract.py:21–23, 44`; absent `docs/visual-replay/interface.md`.

**Finding:**  
Replay exposes video generation and sidecars, not a public image-observation stream. `--no-video` saves rendering inputs; the feature consumer imports `ReplayConfig`, constructs the Gabor patch, and reapplies wheel displacement to regenerate images. It therefore owns replay assembly details. Replay's dependency fingerprint also includes `src/prepare_visual_stim.py`, so changes to downstream CLIP code invalidate direct replay compatibility. The required public contract has no current interface document.

**Expected:**  
Architecture §§7–9 and `docs/dependencies.md` require replay-owned observations usable independently of MP4 and CLIP, with consumers using the public boundary. Feature sampling remains a downstream responsibility.

**Suggested disposition:**  
Wrap / centralize observation generation in replay and document its implemented interface; remove downstream implementation identity from replay's dependency closure.

---

### A06 — Fingerprints do not establish completed generation identity

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Data integrity

**Location:**  
`src/visual_stim_gen.py:268–272, 346–373`; `src/utils/replay_contract.py:129–130, 159–172, 186–207, 210–282`.

**Finding:**  
Contract fingerprints bind implementation/configuration, and record fingerprints bind trial metadata. Neither binds actual frame pixels, sidecar timing/trajectory arrays, or the full generation's trial membership and source identities. Sidecar validation checks internal consistency but not content identity: changing wheel deltas and their matching azimuths leaves the record fingerprint unchanged. The manifest validator checks duplicate IDs without an independent requested-trial set, so removing a trial record does not establish a completeness failure. Source dataset metadata and optional recovered-table fingerprints are useful but do not supply the missing completed-output identity.

**Expected:**  
Architecture §7 requires a completed generation identity binding consumed/produced content, associations, timing, statuses, and provenance, distinct from its reconstruction definition. Requested domain, observation membership, and coverage must remain explicit.

**Suggested disposition:**  
Refactor output assembly to finalize a content-bound generation manifest/completion record, retaining existing structural validation.

---

### A07 — Trial failures can abort publication, while invalid trials report session success

**Priority:** High  
**Confidence:** Confirmed  
**Category:** Pipeline

**Location:**  
`src/visual_stim_gen.py:161–173, 176–184, 313–314, 331–345, 389–413`; `src/utils/sessions.py:55–79` (shared reporting boundary).

**Finding:**  
Only the initial timing/contrast/wheel checks become `InvalidTrial`. Later trial-local failures, including a manifest contrast/side conflict, unsupported sigma, or video-writing error, escape the trial loop and prevent the session from publishing any successfully rendered trials. Parameter resolution also runs for all trials before that loop. Conversely, caught invalid trials receive `valid=False`, but `_generate_session` returns normally and the caller reports the session as completed, even if every trial was invalid. There are no distinct unavailable/invalid/failed outcomes or explicit partial reconstruction result. Publication is staged at session level rather than publishing settled trial artifacts independently.

**Expected:**  
Architecture §8 chooses trial-level failure isolation, explicit outcomes for every requested trial, and partial/failed reconstruction status distinct from accounting completeness. Unexpected interruption must remain unfinished.

**Suggested disposition:**  
Refactor trial outcome handling and staged publication; retain interruption propagation, recoverable staging files, and protection against overwriting completed outputs.

## Conforming areas

- The session generator loads trial and wheel sources through `SessionAccess`, jointly resolves wheel arrays, and records source identities and revision groups.
- Original zero-based trial rows survive skipped trials; recovered parameter tables are checked against the current ALF table when a fingerprint is supplied.
- Contrast zero is valid, contradictory sides are rejected, and movement is not clamped to successful trajectories.
- Frame timestamps are float64 source-session times in the half-open onset/offset interval. Video PTS and source indices are explicit; no arbitrary minimum duration is added.
- Freeze position is interpolated at the event itself, including when no video frame lands on it, and held until offset.
- Parameter units, evidence precedence, fallback provenance, and synthetic phase identity are explicit. Phase depends on seed/EID/original trial ID rather than processing order.
- Video decoding verifies frame count and dimensions before publication. Fresh staging and destination refusal protect completed session outputs.
- Dead-code inspection found active CLI or consumer use of the substantive replay/recovery helpers. Unimplemented state/scene declarations in the contract are not evidence of functioning backends, but no substantive implementation was confidently established as dead. No removal finding is warranted.
