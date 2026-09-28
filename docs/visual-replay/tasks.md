# Visual replay implementation plan

Implement the source-derived replay defined in [spec.md](spec.md) and [architecture.md](architecture.md), using [audit.md](audit.md) as the implementation-gap snapshot. The outcome is a replay-owned stream of mouse-perspective observations and optional persisted artifacts with source timing, original identities, and explicit generation outcomes.

Work stays within `visual-replay` and consumes the public [session-data interface](../session-data/interface.md). Migrating `visual-features` to consume the new boundary is separate downstream work; this plan supplies and verifies the replay side of that boundary. Preserve conforming parameter recovery, identity, timing, and staging behavior where compatible.

Validation belongs within each task: use code inspection, existing checks, direct execution, and manual inspection of representative outputs. Do not write tests or validation infrastructure. Document the actual public contract in `interface.md` after implementation, following the repository interface rules.

## Tasks

### [x] VR01 — Resolve source behavior and reconstruction inputs

**Summary:**

Produce a resolved reconstruction plan from session-data evidence and explicit configuration, following architecture §3. Establish supported historical behavior profiles and their session applicability, effective parameter units/precedence, justified fallbacks, and deterministic synthetic properties. Resolve concrete scene and observation settings before execution; retain source trial-table identity and represent trial-local input failures without discarding other trials. Reuse existing recovery and provenance handling where appropriate. Addresses the preparation aspects of A01, A02, and A07.

**Out of scope:**

Rebuilding historical runtimes, exhaustive version coverage, and changing session-data acquisition or cache policy.

### [x] VR02 — Reconstruct task state and observation timing

**Summary:**

Implement a single cadence-independent trial evaluator and an explicit source-time observation schedule under architecture §§4 and 6. Apply profile-specific visibility, coupling, wheel direction/gain, freeze, and offset semantics, including outward trajectories and exact event-time evaluation. Establish an evidence-appropriate wheel interpolation/gap policy and preserve known blanks, unavailable intervals, timing classification, and requested coverage. Addresses A02 and A03.

**Out of scope:**

Display rendering, feature-extraction sampling, and visual-neural alignment.

### [x] VR03 — Render source-derived display images

**Summary:**

Render display-space images from resolved stimulus state using the selected source behavior for carrier, phase, envelope/aperture, contrast, background, task-to-display mapping, and clipping, as required by architecture §5. Keep movement exclusively in the trial evaluator. Demonstrate that representative side, contrast, and position inputs produce the intended visible output. Addresses A01.

**Out of scope:**

Historical pixel-equivalence certification, physical scene projection, and a general renderer plugin framework.

### [x] VR04 — Project the schematic mouse view

**Summary:**

Project completed display images onto the configured physical screen and fixed mouse-centered camera, preserving aspect ratio and the complete relevant screen area under architecture §5. Record geometry and viewpoint assumptions, expose display and scene images distinctly, and fail explicitly when requested scene output cannot be produced. Make the projection optional in the pipeline. Addresses A04.

**Out of scope:**

Retinal optics, eye tracking, reconstructed room lighting, physical display calibration, and CLIP preprocessing.

### [x] VR05 — Expose replay observations and generation completion

**Summary:**

Expose bounded direct image observations independently of MP4 and CLIP, assembling identity, source timing, image-space/format, validity, coverage, and provenance under architecture §§7–8. Finalize a content-bound generation identity distinct from the reconstruction definition, with every requested trial accounted for and partial/failed reconstruction distinguished from accounting completeness. Isolate trial-local failures throughout preparation and rendering; interruptions must not yield completion. Remove downstream feature implementation from replay's dependency identity and document the implemented public stream/completion contract. Addresses the replay-owned boundary in A05, plus A06 and A07.

**Out of scope:**

Modifying the visual-features consumer, embedding generation, and redesigning other components' reporting interfaces.

### [x] VR06 — Publish and read replay artifacts through the public contract

**Summary:**

Connect replay CLI generation, artifact readers, and optional video output to the same observation records and completion semantics as the direct path. Stage and publish consistent trial images/metadata independently, then publish the generation manifest after request accounting settles; preserve completed generations on failure or interruption. Record any encoded-frame/source-observation mapping without changing scientific time or concealing gaps. Demonstrate readable representative output and public readback preserving identity, timing, statuses, and generation binding; finalize `interface.md` for the implemented CLI and persistence contract. Completes the persisted-output aspects of A05–A07.

**Out of scope:**

Automatic migration of historical replay artifacts, downstream CLIP integration, and additional storage backends beyond the required replay workflow.

## Dependencies

- VR02 depends on VR01.
- VR03 depends on VR02.
- VR04 depends on VR03.
- VR05 depends on VR04.
- VR06 depends on VR05.
