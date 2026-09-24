# VR01: pinned IBL renderer source audit

Audit date: **2026-09-19**. Outcome: **source audit complete; runtime and session compatibility unverified**.

This is the evidence record for [VR01](ibl-source-based-visual-replay-tasks.md#vr01--audit-and-pin-upstream-rendering-sources). The machine-readable [source lock](ibl-renderer-sources.lock.json) records immutable revisions, Git blob hashes for 93 source files, all 56 packages from the historical IBLRIG configuration, and SHA-256 hashes for two downloaded artifacts. It is a source/configuration lock, not a fully restored binary environment.

## Selected sources

| Component | Discovery reference | Immutable commit | Evidence |
| --- | --- | --- | --- |
| IBLRIG | `iblrigv8` branch | `a0a031e2f9969c192767a255781920028034b7b0` | [Selected workflow](https://github.com/int-brain-lab/iblrig/blob/a0a031e2f9969c192767a255781920028034b7b0/visual_stim/GaborIBLTask/Gabor2D.bonsai) |
| BonVision | `v0.9.0` tag | `85992cb492998bfa83e0e6d12d2b37d67df6f2c6` | [Project version and embedded resources](https://github.com/bonvision/BonVision/blob/85992cb492998bfa83e0e6d12d2b37d67df6f2c6/BonVision/BonVision.csproj) |
| Bonsai | `2.6.3` tag | `68f44e7a53c5553631f4d87981a3c748352d4abf` | [Shader project version](https://github.com/bonsai-rx/bonsai/blob/68f44e7a53c5553631f4d87981a3c748352d4abf/Bonsai.Shaders/Bonsai.Shaders.csproj) |

These references were resolved through GitHub's Git/commit API. Subsequent acquisition must use the commits, not the moving branch or tags. File hashes in the lock are **Git blob SHA-1**, which hashes the Git object header plus file bytes; they are not ordinary file SHA-1 checksums. Downloaded archive hashes are ordinary SHA-256 over the archive bytes.

The selected entry point is exactly `visual_stim/GaborIBLTask/Gabor2D.bonsai`. The same directory contains `Gabor2D_clean.bonsai` and `Gabor2D_clean2.5.bonsai`, and the checkout also has a `GaborIBLTask_6.5.2` directory. None is an interchangeable replacement. The lock inventories the selected directory's alternatives for identification; it does not select them for replay.

## Workflow and resource inventory

All paths below are relative to the corresponding pinned repository. Consult the lock for individual blob hashes.

| Source group | Files / resources | Role |
| --- | --- | --- |
| IBLRIG entry point | `visual_stim/GaborIBLTask/Gabor2D.bonsai` | Selected stimulus/control graph; declares workflow builder version 2.6.2 |
| IBLRIG editor state | `Gabor2D.bonsai.layout` in the same directory | Preserve with the reference workflow; editor layout is not the stimulus algorithm |
| IBLRIG transport | `Osc.config` in the same directory | Named `bpod` UDP transport, local port 7110, remote host 127.0.0.1, remote port 0; interpretation/use belongs to VR03 |
| Historical environment | `Bonsai/Bonsai.config`, `Bonsai/install.ps1` | Installed package/assembly configuration and portable bootstrap source |
| Protocol context | `iblrig/base_choice_world.py`, `iblrig/base_choice_world_params.yaml` | Pinned task/controller reference, not automatically session-specific evidence |
| BonVision direct includes | `Primitives/BonVisionResources.bonsai`, `Environment/OrthographicView.bonsai`, `Environment/SphereMapping.bonsai`, `Environment/ViewWindow.bonsai`, `Environment/NormalizedView.bonsai`, `Primitives/DrawQuad.bonsai`, `Primitives/DrawGratings.bonsai` under `BonVision/` | All seven package workflows directly included by the selected IBL graph |
| BonVision mesh resources | `BonVision/Models/Quad.obj`, `BonVision/Models/Plane.obj` | Loaded by the shared resource workflow |
| BonVision implementation support | C# helpers, project/assembly metadata, other embedded workflows | The lock includes all files under `BonVision/`, plus `Extensions.csproj` and `LICENSE`, as a conservative source inventory |
| Bonsai source support | `Bonsai.Shaders/CreateWindow.cs`, `ShaderWindow.cs`, shader/core/OSC/launcher project files, `NuGet.config`, `LICENSE` | Pinned window/runtime source context; whole repository remains pinned by commit |

BonVision includes use assembly-resource identifiers such as `BonVision:Environment.SphereMapping.bonsai`, rather than files alongside the IBL workflow. BonVision's project embeds workflows, models, and shaders into its assembly. Acquiring only `Gabor2D.bonsai` is insufficient.

The shared [BonVisionResources workflow](https://github.com/bonvision/BonVision/blob/85992cb492998bfa83e0e6d12d2b37d67df6f2c6/BonVision/Primitives/BonVisionResources.bonsai) registers this shader resource set under `BonVision/Shaders/`:

- Vertex shaders: `Quad.vert`, `QuadArray.vert`, `Gratings.vert`, `Model.vert`, `TexturedModel.vert`, `MeshMap.vert`, `PerspectiveMap.vert`, `SphereMap.vert`.
- Fragment shaders: `Checkerboard.frag`, `Image.frag`, `Color.frag`, `Gratings.frag`, `Circle.frag`, `Model.frag`, `TexturedModel.frag`, `MeshMap.frag`, `PerspectiveMap.frag`, `SphereMap.frag`, `Gamma.frag`.

All are inventoried, including shared resources that may not be actively drawn. In this revision, SphereMapping refers to the `MeshMap` shader, ViewWindow to `SphereMap`, DrawQuad to `Color`, and DrawGratings to `Gratings`. The existence/loading of `Gamma.frag` does not establish that gamma correction is applied by the selected task. Full data-flow and shader semantics remain VR03 work.

## Historical package versions and compatibility

The authority for the **configured runtime versions** is the pinned [IBLRIG Bonsai.config](https://github.com/int-brain-lab/iblrig/blob/a0a031e2f9969c192767a255781920028034b7b0/Bonsai/Bonsai.config). Its complete 56-entry package list is copied into the lock without resolving or upgrading it.

| Component | Configured version |
| --- | --- |
| Bonsai launcher / Editor / Player | 2.6.3 |
| Bonsai.Core | 2.6.2 |
| Bonsai.Design | 2.6.1 |
| Bonsai.Design.Visualizers | 2.6.2 |
| Bonsai.Osc / Dsp / Scripting / Vision | 2.6.1 |
| Bonsai.System / Windows.Input | 2.6.0 |
| Bonsai.Shaders / Shaders.Design | 0.25.0 |
| Bonsai.Shaders.Rendering | 0.2.1 |
| Bonsai.Numerics | 0.6.0 |
| Bonsai.VR | 0.6.0 |
| Bonsai.Bpod | 0.1.0-ctp1 |
| BonVision | 0.9.0 |
| OpenTK / OpenTK.GLControl | 3.1.0 |
| OpenCV.Net | 3.3.1 |

The workflow builder's `2.6.2` attribute is not the launcher version. The pinned Bonsai shader project explicitly declares `0.25.0`. BonVision's pinned project and downloaded 0.9.0 NuGet manifest both declare .NET Framework 4.6.2 and dependency versions including Shaders 0.24.0, Shaders.Rendering 0.1.0, and Scripting 2.6.0. Those declarations differ from IBLRIG's installed configuration; preserve IBLRIG's exact versions instead of letting a fresh install choose dependencies from minimum requirements.

These are historically co-configured versions, not a claim that this machine has successfully run them. The portable launcher is a bootstrap, not an offline complete package environment. Package availability, remaining package hashes/licenses, native dependency resolution, and capture capability must be recorded during VR02 restore. The correspondence between the BonVision source tag and all resources embedded in its published DLL has not been byte-verified.

One configuration detail to retain for VR02: the package list names `openal.redist` 2.0.7, while native library paths contain `openal.redist.2.0.7.0`. Preserve the original configuration until package restore determines how this path is resolved; do not silently normalize it.

## Downloaded artifacts

Downloaded into ignored `tmp/vr01-source-audit/` for inspection only; no executable was launched and nothing was installed.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| [Bonsai 2.6.3 portable archive](https://github.com/bonsai-rx/bonsai/releases/download/2.6.3/Bonsai.zip) | 935313 | `46fa4d848b5ede89866ddcc2424672e4c7140c07b5c7b7cbcdfdd218f6c541c6` |
| [BonVision 0.9.0 NuGet package](https://api.nuget.org/v3-flatcontainer/bonvision/0.9.0/bonvision.0.9.0.nupkg) | 46942 | `2a0396ba4c78d07df965ab1a32b9d571788fd790116cc20ec134ea04ebc9f9f6` |

The archive contains `Bonsai.exe`, `Bonsai32.exe`, `NuGet.Config`, and empty Extensions/Gallery directories. The pinned IBLRIG installer downloads that same release URL and then launches Bonsai; VR01 deliberately stops before the launch step. These hashes pin the bytes retrieved on the audit date, without claiming an upstream signature or reproducible build.

## Acquisition recipe for subsequent tasks

Run from the ViNED checkout root. Use a **fresh** ignored directory; do not clone into or update an existing historical environment. The following source-checkout recipe is provided for VR02/VR03; clone/checkout commands were not executed during this audit.

```powershell
$ErrorActionPreference = 'Stop'
$sourceLock = Get-Content docs/ibl-renderer-sources.lock.json -Raw | ConvertFrom-Json
$sourceRoot = Join-Path (Get-Location) 'tmp/ibl-renderer-reference'
if (Test-Path -LiteralPath $sourceRoot) { throw 'Choose a fresh source directory' }
New-Item -ItemType Directory -Path $sourceRoot | Out-Null
foreach ($source in $sourceLock.sources) {
    $checkout = Join-Path $sourceRoot $source.id
    git clone --no-checkout $source.repository $checkout
    if ($LASTEXITCODE -ne 0) { throw 'Source clone failed' }
    git -C $checkout checkout --detach $source.commit
    if ($LASTEXITCODE -ne 0) { throw 'Pinned checkout failed' }
    $head = git -C $checkout rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $head -ne $source.commit) { throw 'Commit mismatch' }
    foreach ($file in $source.inventory) {
        $object = git -C $checkout rev-parse "HEAD:$($file.path)"
        if ($LASTEXITCODE -ne 0 -or $object -ne $file.git_blob_sha1) {
            throw "Inventory mismatch: $($source.id)/$($file.path)"
        }
    }
}
```

Git object checks verify repository content, independent of checkout line-ending conversion. For a raw-byte working-tree comparison use a checkout with `core.autocrlf=false` and verify bytes before adapting anything.

Download the binary references into a separate fresh directory and compare against the locked hashes before extraction or execution:

```powershell
$ErrorActionPreference = 'Stop'
$sourceLock = Get-Content docs/ibl-renderer-sources.lock.json -Raw | ConvertFrom-Json
$artifactRoot = Join-Path (Get-Location) 'tmp/ibl-renderer-downloads'
if (Test-Path -LiteralPath $artifactRoot) { throw 'Choose a fresh artifact directory' }
New-Item -ItemType Directory -Path $artifactRoot | Out-Null
foreach ($artifact in $sourceLock.downloaded_artifacts) {
    $destination = Join-Path $artifactRoot $artifact.name
    Invoke-WebRequest -UseBasicParsing $artifact.url -OutFile $destination
    $actual = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
    if ($actual -ne $artifact.sha256) { throw "Artifact mismatch: $($artifact.name)" }
}
```

The URLs and hash procedure above were exercised during the audit. Full source checkouts and runtime restoration were not. Retain a pristine source checkout and make runtime adaptations in a separate working copy. For VR02, extract the verified portable archive into an isolated runtime directory, retain the pinned IBLRIG package configuration, and restore its exact package versions using the historical launcher/package mechanism. Capture the actual restored package artifacts and resolution changes before assessing compatibility. Do not substitute latest packages for unavailable historical ones without a recorded feasibility decision.

Do not execute the full upstream rig installation guide as part of this source acquisition: it also installs Python, camera SDKs, and rig tooling. Determine the renderer's actual hardware/runtime needs in VR02.

## Licenses and attribution

The inspected pinned license files grant MIT permissions:

| Source | Copyright notice | Pinned license |
| --- | --- | --- |
| IBLRIG | Copyright (c) 2022 International Brain Laboratory | [LICENSE](https://github.com/int-brain-lab/iblrig/blob/a0a031e2f9969c192767a255781920028034b7b0/LICENSE) |
| BonVision | Copyright (c) 2019 Aman Saleem | [LICENSE](https://github.com/bonvision/BonVision/blob/85992cb492998bfa83e0e6d12d2b37d67df6f2c6/LICENSE) |
| Bonsai | Copyright (C) 2011-2021 Gonçalo Lopes | [LICENSE](https://github.com/bonsai-rx/bonsai/blob/68f44e7a53c5553631f4d87981a3c748352d4abf/LICENSE) |

Retain each applicable full copyright and permission notice with copies or substantial portions, including source-derived ports. Preserve existing file notices and identify local changes. BonVision's project/package metadata additionally says Copyright © Aman Saleem 2020; preserve that metadata rather than replacing the pinned license's 2019 notice. Its package license URL points to a moving branch, so the pinned license above is the audit reference.

This change inventories sources and packages; it does not vendor implementation code or redistribute their binaries. If later tasks copy them into ViNED, include the applicable complete notices and update [LICENSING.md](../LICENSING.md) to identify those contributions. Other packages and native components retain their individual licenses; the three repository MIT licenses are not a blanket license for the 56-package environment. Preserve NEDS attribution and the existing scope of original ViNED contributions.

## Local dependency review and session applicability

The existing Windows and Linux constraint files govern ViNED's Python 3.10 environment. No pip requirements or constraints are changed. The historical configuration uses .NET Framework assemblies (`net462` and launcher `net472`), OpenTK/OpenCV.Net, Windows input, and native-library paths. These are separate from ViNED's Python OpenCV/NumPy stack; `OpenCV.Net 3.3.1` is not a requested downgrade of Python OpenCV. Keep historical Bonsai packages, any IBLRIG Python environment, and ViNED `.venv` isolated. Linux renderer compatibility has not been established.

The reference workflow is selected because the specification names it. **No configured EID has yet been matched to this commit or these package versions.** All per-session matches are unknown pending VR04. The additional workflow variants demonstrate that file choice can matter, but do not prove which variant any session used. This audit does not claim a measured rendering difference between them.

VR04 must reconcile session/task version evidence with this reference, including parameter defaults and any display/capture adaptations. Keep the reference lock immutable for this generation; use a separately identified revision if a session requires another source. Monitor scaling, calibration, actual frame timing, and exact original sensory input remain unverified.

## Verification performed and handoff

- Retrieved complete, non-truncated Git trees for all three pinned revisions; recorded 13 IBLRIG, 72 BonVision, and 8 Bonsai file entries.
- Inspected the entry workflow, its seven BonVision include references, shared shader/model resources, package configuration, project versions, installer, OSC settings, and pinned license files.
- Independently computed Git blob hashes from raw bytes for the IBL workflow, IBL package configuration, BonVision project, and Bonsai shader project; all four matched the inventory.
- Downloaded both listed archives, computed their SHA-256 hashes, inspected portable ZIP entries, and read the BonVision NuGet manifest without executing assemblies.
- Compared the historical dependency families with the local Windows/Linux Python constraints; no application dependency changes are needed for VR01.
- Runtime launch, dependency restoration, graphics output, source-to-DLL resource equivalence, and session matches were not verified. These remain explicit VR02/VR03/VR04 responsibilities.

VR01 is complete as a source audit. VR02 can now attempt the exact historical environment, and VR03 can inspect a pinned implementation instead of moving upstream files.
