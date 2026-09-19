import argparse
import logging
import os
import sys
import json
import tempfile
from pathlib import Path
from utils.paths import visual_dir, REPO_ROOT
from utils.provenance import file_hash, fingerprint, source_hashes, split_trials, write_json
from utils.sessions import add_session_arguments, select_sessions, run_sessions, SkipSession

import numpy as np
import pandas as pd
from one.api import ONE

from datasets import DatasetDict, DatasetInfo
from utils.dataset_utils import create_dataset, upload_dataset
from utils.ibl_data_utils import (
    align_data,
    bin_behaviors,
    bin_spiking_data,
    list_brain_regions,
    prepare_data,
    select_brain_regions,
)

logging.basicConfig(level=logging.INFO)


# ------
# SET UP
# ------
ap = argparse.ArgumentParser()
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--huggingface_org", type=str, default="FreyBence")
ap.add_argument("--use_lfp", action="store_false")
add_session_arguments(ap)
ap.add_argument("--n_workers", type=int, default=1)
ap.add_argument("--split-seed", type=int, default=42)
args = ap.parse_args()

eids = select_sessions(args.eid, args.eids_file, args.n_sessions)

params = {
    "interval_len": 2,
    "binsize": 0.02,
    "single_region": False,
    "align_time": 'stimOn_times',
    "time_window": (-.5, 1.5),
    "fr_thresh": 0.2
}

beh_names = ["vision-clip"]

DYNAMIC_VARS = ["vision-clip"]

# ---------------
# PREPROCESS DATA
# ---------------
one = ONE(
    base_url="https://openalyx.internationalbrainlab.org",
    password="international",
    silent=True,
    cache_dir=args.base_path
)

def prepare_session(eid):
    destination = Path(args.base_path) / f"{eid}_aligned"
    if destination.exists():
        raise FileExistsError(f"Use a fresh data root; aligned dataset exists: {destination}")
    feature_path = visual_dir() / f"{eid}_visual_clip.npz"
    feature_sha256 = file_hash(feature_path)
    with np.load(feature_path, allow_pickle=False) as features:
        if "provenance" not in features.files:
            raise ValueError("Regenerate CLIP features with revision provenance before preparing data")
        visual_provenance = json.loads(features["provenance"].item())

    # if os.path.exists(f"{args.base_path}/{eid}_aligned"):
    #     logging.info(f"The dataset {eid}_aligned already exists.")
    #     continue

    logging.info(f"EID {eid}")

    neural_dict, behave_dict, meta_dict, trials_dict, _ = prepare_data(
        one, eid, params, n_workers=args.n_workers
    )

    if neural_dict is None:
        raise SkipSession("Missing spike data")

    regions, beryl_reg = list_brain_regions(neural_dict, **params)
    region_cluster_ids = select_brain_regions(neural_dict, beryl_reg, regions, **params)

    bin_spikes, clusters_used_in_bins = bin_spiking_data(
        region_cluster_ids,
        neural_dict,
        trials_df=trials_dict["trials_df"],
        n_workers=args.n_workers,
        **params
    )

    logging.info(f"Binned Spike Data: {bin_spikes.shape}")

    # Keep responsive neurons
    mean_fr = bin_spikes.sum(1).mean(0) / params["interval_len"]
    keep_unit_idxs = np.argwhere(mean_fr > 1/params["fr_thresh"]).flatten()
    bin_spikes = bin_spikes[..., keep_unit_idxs]
    logging.info(
        f"# Responsive Units: {bin_spikes.shape[-1]} / {len(mean_fr)}"
    )

    meta_dict["cluster_regions"] = [meta_dict["cluster_regions"][idx] for idx in keep_unit_idxs]
    meta_dict["cluster_channels"] = [meta_dict["cluster_channels"][idx] for idx in keep_unit_idxs]
    meta_dict["cluster_depths"] = [meta_dict["cluster_depths"][idx] for idx in keep_unit_idxs]
    meta_dict["good_clusters"] = [meta_dict["good_clusters"][idx] for idx in keep_unit_idxs]
    meta_dict["uuids"] = [meta_dict["uuids"][idx] for idx in keep_unit_idxs]
    # meta_dict["cluster_qc"] = {
    #     k: np.asarray(v)[keep_unit_idxs].tolist() for k, v in meta_dict["cluster_qc"].items()
    # }

    bin_beh, beh_mask = bin_behaviors(
        one,
        eid,
        DYNAMIC_VARS,
        trials_df=trials_dict["trials_df"],
        allow_nans=True,
        n_workers=args.n_workers,
        **params,
    )

    if args.use_lfp == False:
        from utils.preprocess_lfp import featurize_lfp, prepare_lfp

        lfp_prec = prepare_lfp(one, eid, dead_channel_threshold=0., **params)
        all_psd = featurize_lfp(
            lfp_prec, bin_size=int(params["interval_len"]/params["binsize"])
        )
        bin_lfp = []
        for lfp_band in all_psd.values():
            bin_lfp.append(lfp_band)
        bin_lfp = np.concatenate(bin_lfp, -1)
        logging.info(f"Binned LFP Data: {bin_lfp.shape}")
    else:
        bin_lfp = None

    try:
        align_bin_spikes, align_bin_beh, align_bin_lfp, target_mask, bad_trial_idxs = align_data(
            bin_spikes,
            bin_beh,
            bin_lfp,
            list(bin_beh.keys()),
            trials_dict["trials_mask"],
            behavior_masks=beh_mask,
            trial_index=trials_dict["trials_df"].index,
        )
    except ValueError as e:
        raise ValueError(f"Alignment failed for {eid}: {e}") from e

    # Data partition (train: 0.7 val: 0.1 test: 0.2)
    num_trials = len(align_bin_spikes)

    trial_mask = np.array(target_mask).astype(bool).tolist()
    rejected_trial_ids = trials_dict["trials_df"].loc[
        ~np.asarray(trial_mask), "original_trial_id"].to_numpy(dtype=np.int64).tolist()
    trials_dict['trials_df'] = trials_dict['trials_df'][trial_mask]
    intervals = np.vstack([
        trials_dict['trials_df'][params["align_time"]] + params["time_window"][0],
        trials_dict['trials_df'][params["align_time"]] + params["time_window"][1]
    ]).T
    original_trial_ids = trials_dict["trials_df"]["original_trial_id"].to_numpy(dtype=np.int64)

    splits, split_metadata = split_trials(eid, original_trial_ids, intervals, args.split_seed)
    train_idxs, val_idxs, test_idxs = (splits[name] for name in ("train", "val", "test"))
    if file_hash(feature_path) != feature_sha256:
        raise ValueError("Visual features changed during alignment; retry with stable inputs")
    provenance = dict(schema_version=1, eid=eid, subject=str(meta_dict["subject"]),
                      params=params, split=split_metadata,
                      visual_sha256=feature_sha256, visual=visual_provenance,
                      sources=source_hashes("src/prepare_data.py", "src/utils/ibl_data_utils.py",
                                            "src/utils/dataset_utils.py", "src/utils/visual_data.py",
                                            "src/utils/provenance.py"),
                      selection=dict(valid_trial_ids=original_trial_ids.tolist(),
                                     rejected_trial_ids=rejected_trial_ids,
                                     intervals=intervals.tolist(),
                                     reaction_time_seconds=[0., 10.], exclude_nochoice=True,
                                     max_trial_length_seconds=10.,
                                     missing_events_excluded=["stimOn_times", "choice", "feedback_times",
                                                              "probabilityLeft", "firstMovement_times", "feedbackType"],
                                     visual_policy="at least one valid aligned visual bin",
                                     firing_rate_policy="session-level QC before splitting",
                                     firing_rate_threshold_hz=1/params["fr_thresh"]),
                      neuron_order={key: [str(value) for value in meta_dict[key]] for key in
                                    ("uuids", "cluster_regions")})
    provenance["fingerprint"] = fingerprint(provenance)

    train_beh, val_beh, test_beh = {}, {}, {}
    for beh in align_bin_beh.keys():
        train_beh.update({beh: align_bin_beh[beh][train_idxs]})
        val_beh.update({beh: align_bin_beh[beh][val_idxs]})
        test_beh.update({beh: align_bin_beh[beh][test_idxs]})

    train_dataset = create_dataset(
        align_bin_spikes[train_idxs],
        eid,
        params,
        meta_data=meta_dict,
        binned_behaviors=train_beh,
        trial_ids=original_trial_ids[train_idxs],
        intervals=intervals[train_idxs],
        binned_lfp=None if align_bin_lfp is None else align_bin_lfp[train_idxs]
    )

    val_dataset = create_dataset(
        align_bin_spikes[val_idxs],
        eid,
        params,
        meta_data=meta_dict,
        binned_behaviors=val_beh,
        trial_ids=original_trial_ids[val_idxs],
        intervals=intervals[val_idxs],
        binned_lfp=None if align_bin_lfp is None else align_bin_lfp[val_idxs]
    )

    test_dataset = create_dataset(
        align_bin_spikes[test_idxs],
        eid,
        params,
        meta_data=meta_dict,
        binned_behaviors=test_beh,
        trial_ids=original_trial_ids[test_idxs],
        intervals=intervals[test_idxs],
        binned_lfp=None if align_bin_lfp is None else align_bin_lfp[test_idxs]
    )

    dataset = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )
    logging.info(dataset)

    # upload_dataset(dataset, org=args.huggingface_org, eid=f"{eid}_aligned")
    for name in dataset:
        dataset[name] = dataset[name].add_column("split", [name] * len(dataset[name]))
        dataset[name] = dataset[name].add_column("provenance_id", [provenance["fingerprint"]] * len(dataset[name]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{eid}-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "dataset"
        dataset.save_to_disk(str(staging))
        write_json(staging / "provenance.json", provenance)
        staging.rename(destination)

    logging.info(f"Downloaded EID: {eid}")


run_sessions(eids, prepare_session, "aligned-data")

# Report session and subject overlap separately from within-session trial splits.
groups = {}
for name in ("train", "test"):
    selected = select_sessions(eids_file=REPO_ROOT / f"data/{name}_eids.txt")
    subjects, unknown = set(), []
    for eid in selected:
        path = Path(args.base_path) / f"{eid}_aligned" / "provenance.json"
        if path.exists():
            subjects.add(json.loads(path.read_text(encoding="utf-8"))["subject"])
        else:
            unknown.append(eid)
    groups[name] = dict(eids=selected, subjects=sorted(subjects), unverified_eids=unknown)
write_json(Path(args.base_path) / "split_independence.json", dict(
    groups=groups, shared_sessions=sorted(set(groups["train"]["eids"]) & set(groups["test"]["eids"])),
    shared_subjects=sorted(set(groups["train"]["subjects"]) & set(groups["test"]["subjects"])),
    complete=not any(g["unverified_eids"] for g in groups.values())))
