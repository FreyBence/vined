# E2 controlled spatial CLIP comparison

Implement the accepted controlled spatial CLIP comparison. Preserve historical experiments and authoritative design documents.

### [x] T1 — Publish common-pass frozen A/B/G source features

**Summary:** Extract and qualify pinned pooled CLS, normalized 2×2 raw patch means, and the global-patch control; publish three independently readable provenance-bound artifacts from bounded common inference.

**Out of scope:** Replay changes, CLIP training, full patch persistence, and additional representations.

### [x] T2 — Publish exactly matched alignment and dataset generations

**Summary:** Reuse the reference neural arrays, source identities, interpolation brackets, complete bins, anatomical ordering, and all 312 frozen split memberships for the three representation generations.

**Out of scope:** Neural regeneration, resplitting, changed alignment, and loader redesign.

### [ ] T3 — Run and report the controlled screening comparison

**Summary:** Train fresh strict H12/D0/S12 A/B/G models with controlled shared initialization and the unchanged 50-epoch protocol; freeze validation decisions before final test evaluation, report paired metrics, provenance, resources, and manual verification, and update affected implementation interfaces.

**Out of scope:** E0/E1/E1b changes, E3, tuning, convergence studies, automated tests, and CI.

## Fixed dependencies

- T2 depends on T1.
- T3 depends on T2.
