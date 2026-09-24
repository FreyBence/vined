# VR02: original-renderer feasibility decision

Date: **2026-09-19**. Status: **assessment complete; original hardware-free capture not demonstrated**.

**Decision: select VR09, the source-derived Python production path, for the current local environment.** VR08 is not selected. This is a decision based on the failed minimal launch and the remaining hardware/capture adaptations, not proof that Bonsai cannot be adapted to run elsewhere. Continue to use the pinned original implementation as the rendering authority. A validated Python renderer still requires original-workflow reference frames in VR10.

Evidence: [source audit](ibl-renderer-source-audit.md), [immutable source lock](ibl-renderer-sources.lock.json), and [feasibility evidence record](ibl-renderer-feasibility-evidence.json). The latter records package hashes, local diagnostic-file hashes, environment observations, attempts, and explicit unverified results. Runtime downloads and logs remain in ignored `tmp/vr02-feasibility/`.

## What ran and what failed

| Attempt | Action | Observed result |
| --- | --- | --- |
| 1: historical restore | Extracted the hash-pinned Bonsai 2.6.3 portable archive, supplied the original IBLRIG `Bonsai.config`, and launched the bootstrap with `--no-editor` | Restored packages until the configured feeds could not find `Bonsai.PointGrey 2.2.0-ivo2` |
| 2: restore without unused camera package | Removed PointGrey/FlyCapture configuration entries from a runtime copy; retained all other configured versions; reran the bootstrap | Restored all 55 retained configured packages; stderr empty; package list comparison found only the deliberate PointGrey removal |
| 3: selected workflow launch | Ran the unchanged pinned `Gabor2D.bonsai` graph with `--no-editor --no-boot`, primary-display selection, and local CSV destinations | `System.IO.IOException` while opening absent `COM7` in `Bonsai.Bpod.Encoder.Generate`; process exited. OpenTK subsequently raised `System.NullReferenceException` during window disposal |

The exact restore diagnostic was: `The package 'Bonsai.PointGrey 2.2.0-ivo2' could not be found.` This means unavailable through the configured restore attempt, not proven absent from every archive or package feed. PointGrey is not referenced by the selected stimulus graph, so that restore failure alone was not used to reject original rendering.

The workflow then reached execution and attempted to open the rotary encoder's serial port. The recorded port enumeration returned no serial ports. No physical wheel or task device was attached, and no hardware outputs were disabled or simulated. The COM7 failure is the first observed execution blocker; the later OpenTK disposal exception is a separate cleanup failure, not evidence of a successful graphics context or stimulus rendering.

CSV initialization produced empty event/position/trial files and one initial record each in the screen-position and synchronization files. These are state/logging observations, not captured frames. No numeric process exit code was retained; the evidence records observed process exit and diagnostics instead. The launched processes were no longer running at the end of inspection.

## Environment and restored artifacts

| Item | Observation |
| --- | --- |
| Runtime | Windows PowerShell; installed .NET Framework registry version `4.8.09221`, release `533509` |
| Detected graphics adapters | Intel Iris Xe Graphics, driver `31.0.101.4502`; NVIDIA GeForce RTX 3060 Laptop GPU, driver `32.0.15.9597` |
| Displays | One primary display; graphics-controller report 1920 x 1080; Windows Forms bounds 1536 x 864 |
| Actual OpenGL device/framebuffer | Not measured; monitor API dimensions are not framebuffer dimensions |
| Selected source render request | 640 x 480, fullscreen, VSync on, target 60 Hz; original display index 1 changed to 0 through an exposed property |
| Serial devices | No enumerated ports; original graph requests COM7 |
| Package restore | 55 retained configured versions, plus four OpenCV overlay dependencies; 59 downloaded package archives hashed |
| Captured stimulus frames | Zero |

The differently reported display dimensions may reflect desktop scaling; no scaling measurement was made. They must not be used to claim the renderer produced either resolution.

The four additional archives were `OpenCV.ffmpeg.overlay-Win32_v110`, `OpenCV.ffmpeg.overlay-x64_v110`, `OpenCV.overlay-Win32_v110_Release`, and `OpenCV.overlay-x64_v110_Release`, all version 2.4.8. They were resolved during restore, not upgrades to the 55 retained package versions. The `openal.redist` 2.0.7 package restored into `openal.redist.2.0.7.0`, matching the historical native-library directory naming noted in VR01.

The MyGet-restored BonVision 0.9.0 archive has the same SHA-256 as the NuGet download pinned in VR01. Every restored archive's hash and advertised license metadata is retained in the evidence JSON. License URLs may be mutable or unavailable; collecting package metadata is not a completed redistribution-license review. No package binaries are added to tracked source, and no external machine-wide installer was run. No Python dependencies or platform constraints were changed.

## Adaptations actually applied

1. Prepared a portable runtime under the ignored workspace directory, preserving the original downloaded workflow and configuration alongside it.
2. Removed entries containing `PointGrey` or `FlyCapture` from the runtime package/assembly/native-path configuration after the first restore failed. No stimulus package version was changed. Saved this pre-restore variant as `Bonsai-no-camera.config`.
3. Passed `--property:Stim.DisplayIndex=0` to select the primary display. The original requested fullscreen state, 640 x 480 dimensions, VSync, and target cadence were not changed. The resulting framebuffer dimensions were never observed.
4. Redirected the five exposed CSV destinations into the feasibility directory to avoid writes to the original rig-specific `C:\iblrig_data\...` locations.
5. Started the launcher with `-WindowStyle Hidden` and diagnostic stdout/stderr redirection. This does not establish that its OpenGL child window was hidden, nor that hidden/offscreen capture works.

The workflow graph, shaders, geometry, OSC transport, and COM7 encoder remained unchanged. No desktop screenshot was substituted for a framebuffer capture. No hardware-free graph, virtual serial device, custom capture harness, or Python rendering fallback was executed in this task.

## Input and capture feasibility

The selected workflow statically exposes OSC input on the `bpod` UDP transport (local port 7110). Its input graph includes `/p` (integer position), `/c` (float contrast), `/t` (integer trial), `/f`, `/a`, `/s`, `/g`, `/h` (float stimulus parameters), and `/re` (integer task event). Graph labels identify event 8 as showing the stimulus and event 1 as hiding it. This is an interface inspection, not verification of timing, units, or a replay protocol; VR03 must trace those semantics before using them.

| Required observation | Result |
| --- | --- |
| Controlled blank input | Not reached: launch failed during hardware initialization before a controlled input/capture sequence could run |
| Controlled visible stimulus | Not reached for the same reason |
| Delivery of OSC commands to a running renderer | Not demonstrated; no controlled packets were sent after the failed launch |
| Lossless framebuffer access | A source operator exists; not exercised successfully |
| Observed render dimensions / pixel format | Not measured |
| Complete captured frame sequence | Not demonstrated |
| Repeatable state-to-frame correspondence | Not demonstrated |
| Reference-only capture | Not demonstrated; remains an outstanding requirement for VR10 |

The pinned [ReadPixels operator](https://github.com/bonsai-rx/bonsai/blob/68f44e7a53c5553631f4d87981a3c748352d4abf/Bonsai.Shaders/ReadPixels.cs) calls `GL.ReadPixels`, constructs a three-channel 8-bit image, and flips it vertically. With no input it uses one update-frame event. That establishes a potential framebuffer-read mechanism, not a complete capture pipeline.

The pinned [ShaderWindow render loop](https://github.com/bonsai-rx/bonsai/blob/68f44e7a53c5553631f4d87981a3c748352d4abf/Bonsai.Shaders/ShaderWindow.cs) emits render-frame callbacks before dispatching queued shaders and swapping buffers. Attaching a read to that callback without checking ordering/buffer selection would not establish that the pixels correspond to the requested stimulus state. The IBL graph has CSV writers but no ReadPixels/image/video capture stage. A reliable adapter must define the capture point and prove ordering, completeness, and state identity.

## Decision boundaries and handoff

The unmodified task graph did not run without hardware. Removing its encoder, supplying equivalent subject/state streams, and introducing correctly ordered framebuffer capture would require workflow adaptations beyond the minimal original launch performed here. Their feasibility has not been disproved. The current evidence does not meet the VR08 selection criterion of repeatable input and complete capture, so VR09 is selected explicitly for this environment.

- **VR03:** Trace the pinned original control/rendering semantics, including encoder subjects and the shader execution order. The local restore is available as an inspection resource.
- **VR08:** Not applicable under this recorded selection. Reopen the decision if an adapted original workflow can demonstrate controlled blank/visible frames and complete, repeatable state-to-frame capture without hardware.
- **VR09:** Selected but pending its existing prerequisites. Implement from the pinned sources; do not describe its output as source-faithful until VR10 passes.
- **VR10:** Original reference images are still required. A separate hardware-free original-workflow adapter, a suitable rig environment, or traceable captures from that environment are possible routes, but none was demonstrated here. Python output cannot serve as its own original reference.

This decision is not an automatic fallback policy. A later failure of any explicitly selected backend must surface as a failure; changing the production choice requires an updated decision record. No claims of source-renderer fidelity or exact session reconstruction follow from completing VR02.

## Reproducing the attempts

Use a fresh ignored directory, for example `tmp/vr02-retry`, and the VR01 acquisition guide. Verify the source lock's Bonsai ZIP hash before extracting into `runtime/`. Download `Gabor2D.bonsai`, `Osc.config`, and `Bonsai.config` from the pinned IBLRIG commit into the retry directory. Preserve those originals. This task's diagnostics are under `tmp/vr02-feasibility/`; use distinct paths to avoid replacing them.

From the retry directory, copy the original configuration into the runtime and run the historical bootstrap. These are manual operational commands, not a test suite or automated renderer qualification:

```powershell
$ErrorActionPreference = 'Stop'
$attemptRoot = (Get-Location).Path
$runtimeRoot = Join-Path $attemptRoot 'runtime'
Copy-Item -LiteralPath './Bonsai.config' -Destination './runtime/Bonsai.config'
$process = Start-Process -FilePath (Join-Path $runtimeRoot 'Bonsai.exe') `
    -ArgumentList '--no-editor' -WorkingDirectory $runtimeRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $attemptRoot 'bootstrap.stdout.log') `
    -RedirectStandardError (Join-Path $attemptRoot 'bootstrap.stderr.log') -PassThru
$process.Id
$process.WaitForExit(20000)
```

If the wait returns false, retain the process ID and inspect its logs until it exits before changing configuration or launching another restore. Do not treat a timeout as restore failure or success. During this run, the first attempt terminated with the PointGrey error above.

To reproduce the exact configuration adaptation, after that process exits:

```powershell
[xml]$runtimeConfig = Get-Content './Bonsai.config'
$cameraEntries = @($runtimeConfig.SelectNodes('//*[@id or @assemblyName or @path or @location]') |
    Where-Object { $_.OuterXml -match 'PointGrey|FlyCapture' })
foreach ($entry in $cameraEntries) { $entry.ParentNode.RemoveChild($entry) | Out-Null }
$runtimeConfig.Save((Join-Path $attemptRoot 'Bonsai-no-camera.config'))
$runtimeConfig.Save((Join-Path $runtimeRoot 'Bonsai.config'))
```

Repeat the bootstrap command using distinct `restore-no-camera.stdout.log` and `restore-no-camera.stderr.log` destinations. Wait for it to exit and compare the restored configuration against the original; this audit found only the PointGrey removal. Package downloads came from the portable archive's configured feeds, and the exact bytes observed here are recorded in the evidence JSON. Feed availability on another date is not guaranteed.

After restore, launch with the same arguments used in this assessment:

```powershell
$launchArguments = @(
    '..\Gabor2D.bonsai', '--no-editor', '--no-boot',
    '--property:Stim.DisplayIndex=0',
    '--property:Stim.FileNameTrialInfo=..\trial-info.ssv',
    '--property:Stim.FileNameEvents=..\events.ssv',
    '--property:Stim.FileNamePositions=..\positions.ssv',
    '--property:Stim.FileNameStimPositionScreen=..\screen.csv',
    '--property:Stim.FileNameSyncSquareUpdate=..\sync.csv'
)
$process = Start-Process -FilePath (Join-Path $runtimeRoot 'Bonsai.exe') `
    -ArgumentList $launchArguments -WorkingDirectory $runtimeRoot -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $attemptRoot 'original-launch.stdout.log') `
    -RedirectStandardError (Join-Path $attemptRoot 'original-launch.stderr.log') -PassThru
$process.Id
$process.WaitForExit(15000)
```

On this machine the process exited with the COM7 error and subsequent OpenTK exception. If another environment reaches a running renderer, do not infer successful capture from an open window or CSV records: controlled input, framebuffer geometry, capture ordering, and frame completeness still need evidence.

## Completion record

VR02's permitted failed-environment outcome is complete: the historical restore and minimal launch were attempted, the recoverable camera-package issue was addressed, the execution blocker and all applied adaptations were recorded, and the production/reference decisions are explicit. Blank/visible capture acceptance was **not** achieved; those observations remain marked unverified rather than counted as passing checks. No tests or CI/CD configuration were added.
