"""Small, pinned behavioral references, not automatic session-version guesses.

The source graph references BonVision primitives; implementation of those
primitives belongs to the display renderer, not parameter resolution.
"""
from copy import deepcopy

COMMIT = "a0a031e2f9969c192767a255781920028034b7b0"
ROOT = f"https://github.com/int-brain-lab/iblrig/blob/{COMMIT}/"
PROFILE_ID = "iblrig-a0a031e2-choice-world"
PARAMETERS = ROOT + "iblrig/base_choice_world_params.yaml"
WORKFLOW = ROOT + "visual_stim/GaborIBLTask/Gabor2D.bonsai"
BONVISION_COMMIT = "85992cb492998bfa83e0e6d12d2b37d67df6f2c6"
BONVISION_ROOT = f"https://github.com/bonvision/BonVision/blob/{BONVISION_COMMIT}/BonVision/"


def behavior_profile(profile_id):
    if profile_id != PROFILE_ID:
        raise ValueError(f"Unsupported replay behavior profile: {profile_id}")
    return deepcopy(dict(
        id=PROFILE_ID, reference_commit=COMMIT,
        references=dict(task=ROOT + "iblrig/base_choice_world.py",
                        transport=ROOT + "iblrig/base_tasks.py",
                        parameters=PARAMETERS, display=WORKFLOW,
                        grating_shader=BONVISION_ROOT + "Shaders/Gratings.frag",
                        grating_workflow=BONVISION_ROOT + "Primitives/DrawGratings.bonsai",
                        phase_property=BONVISION_ROOT + "AngleProperty.cs",
                        sphere_grid=BONVISION_ROOT + "CreateSphereGrid.cs",
                        sphere_mapping=BONVISION_ROOT + "Environment/SphereMapping.bonsai",
                        view_window=BONVISION_ROOT + "Environment/ViewWindow.bonsai",
                        view_shader=BONVISION_ROOT + "Shaders/SphereMap.vert",
                        blend_state=BONVISION_ROOT + "Primitives/BonVisionResources.bonsai"),
        task=dict(onset="show stationary stimulus", closed_loop="enable wheel coupling",
                  reward="freeze_at_center", error="freeze_in_place", no_go="hide",
                  offset="hide", task_position_unit="deg",
                  simultaneous_event_order=["onset", "closed_loop", "freeze", "offset"],
                  wheel_unit="rad", gain_unit="deg/mm",
                  reversal="negate transmitted gain once when STIM_REVERSE is true",
                  angle_wrap="(position + 180) % 360 - 180"),
        display=dict(workflow="GaborIBLTask/Gabor2D.bonsai",
                     projection="BonVision Environment.ViewWindow; NormalizedView is used for the sync square",
                     carrier="BonVision Primitives.DrawGratings, SquareWave=false",
                     spatial_frequency="task cycles/degree passed to SpatialFrequency",
                     phase="dynamic task radians / (2*pi), subtracted from zero accumulated temporal phase",
                     envelope="StimSize = stim_sigma**2; Radius = Aperture = StimSize / VisualSpan",
                     extent="ExtentX = ExtentY = VisualSpan",
                     visual_span="degrees(2 * atan(ScreenWidth / (2 * abs(TranslationZ))))",
                     orientation="DrawGratings.Angle is fixed at zero; logged stim_angle is not connected",
                     opacity="zero when contrast <= 0.01, otherwise one",
                     background="window ClearColor=Gray",
                     temporal_frequency=0, location_y=0,
                     shader_reference=dict(commit=BONVISION_COMMIT, release="v0.9.0",
                         applicability="Pinned functional reference; session-installed package version is unknown"),
                     renderer=dict(id="source_derived_angular_gabor", version=1,
                         background_rgb=[128, 128, 128], effective_orientation_deg=0.0,
                         phase_sign=-1.0, opacity_threshold=0.01, sigma_power=2,
                         projection="analytic inverse SphereMapping/ViewWindow at pixel centers",
                         blending="SrcAlpha/OneMinusSrcAlpha to RGBA texture, then RGB cubemap",
                         sampling="continuous shader evaluation; final RGB8 rounding only",
                         sync_marker="omitted: source toggle parity is not established by resolved task state")),
        nominal_parameters=dict(wheel_radius_mm=31.0, gain_deg_per_mm=4.0,
                                spatial_frequency_cpd=0.1, sigma_deg=7.0,
                                orientation_deg=0.0),
        nominal_initial_magnitude_deg=35.0,
        nominal_display_geometry=dict(width=20.0, height=15.0, distance=7.0,
                                      unit="source ViewWindow units"),
        limitations=["Reference commit is not evidence of session applicability",
                     "BonVision package revision and physical calibration are not recovered",
                     "ALF wheel sign and event proxies require explicit session configuration"],
    ))
