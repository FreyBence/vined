from utils.paths import visual_dir as get_visual_dir
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from brainbox.io.one import SpikeSortingLoader, _channels_alf2bunch
from utils.sessions import SkipSession
from brainbox.population.decode import get_spike_counts_in_bins
from iblatlas.regions import BrainRegions
from iblutil.numerical import bincount2D, ismember
from tqdm import tqdm

from utils.visual_data import load_archive, resample_features, validate_ids

def load_spiking_data(access, pid, compute_metrics=False, qc=None, **kwargs):
    eid = kwargs.pop("eid", "")
    pname = kwargs.pop("pname", "")
    sampling_freq = 30_000
    result = access.load_spike_sorting(eid, pid=pid, pname=pname or None)
    spikes, clusters, channels = result.spikes, result.clusters, result.channels
    # Channel interpretation and cluster enrichment remain neural processing.
    channels = _channels_alf2bunch(channels, brain_regions=BrainRegions())
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=compute_metrics
    )
    if clusters_labeled is None:
        return None, None, None
    else:
        clusters_labeled = clusters_labeled.to_df()
    clusters_labeled.attrs["source"] = json.loads(json.dumps(asdict(result.source), default=str))
    
    if qc is None:
        return spikes, clusters_labeled, sampling_freq
    else:
        iok = clusters_labeled["label"] >= qc
        selected_clusters = clusters_labeled[iok]
        spike_idx, ib = ismember(spikes["clusters"], selected_clusters.index)
        selected_clusters.reset_index(drop=True, inplace=True)
        selected_spikes = {k: v[spike_idx] for k, v in spikes.items()}
        selected_spikes["clusters"] = selected_clusters.index[ib].astype(np.int32)
        return selected_spikes, selected_clusters, sampling_freq


def merge_probes(spikes_list, clusters_list):
    assert (len(clusters_list) == len(spikes_list)), \
        "clusters_list and spikes_list must have the same length"
    assert all([isinstance(s, dict) for s in spikes_list]), \
        "spikes_list must contain only dictionaries"
    assert all([isinstance(c, pd.DataFrame) for c in clusters_list]), \
        "clusters_list must contain only pd.DataFrames"

    merged_spikes, merged_clusters = [], []
    cluster_max = 0

    for clusters, spikes in zip(clusters_list, spikes_list):
        spikes["clusters"] += cluster_max
        cluster_max += len(clusters)
        merged_spikes.append(spikes)
        merged_clusters.append(clusters)
        
    merged_clusters = pd.concat(merged_clusters, ignore_index=True)
    merged_spikes = {
        k: np.concatenate([s[k] for s in merged_spikes]) for k in merged_spikes[0].keys()
    }
    sort_idx = np.argsort(merged_spikes["times"], kind="stable")
    merged_spikes = {k: v[sort_idx] for k, v in merged_spikes.items()}
    return merged_spikes, merged_clusters


def load_trials_and_mask(
    access,
    eid, 
    min_rt=0.0, 
    max_rt=10., 
    nan_exclude="default", 
    min_trial_len=None,
    max_trial_len=10, 
    exclude_unbiased=False, 
    exclude_nochoice=True, 
    trials=None,
): 
    if nan_exclude == "default":
        nan_exclude = [
            "stimOn_times",
            "choice",
            "feedback_times",
            "probabilityLeft",
            "firstMovement_times",
            "feedbackType"
        ]

    if trials is None:
        trials = access.load_trials(eid).data

    if min_rt is not None:
        query = f"(firstMovement_times - stimOn_times < {min_rt})"
    else:
        query = ""
    if max_rt is not None:
        query += f" | (firstMovement_times - stimOn_times > {max_rt})"
    if min_trial_len is not None:
        query += f" | (feedback_times - goCue_times < {min_trial_len})"
    if max_trial_len is not None:
        query += f" | (feedback_times - goCue_times > {max_trial_len})"
    for event in nan_exclude:
        query += f" | {event}.isnull()"
    if exclude_unbiased:
        query += " | (probabilityLeft == 0.5)"
    if exclude_nochoice:
        query += " | (choice == 0)"
    if min_rt is None:
        query = query[3:]

    mask = ~trials.eval(query)
    return trials, mask


def list_brain_regions(neural_dict, **kwargs):
    brainreg = BrainRegions()
    beryl_reg = brainreg.acronym2acronym(neural_dict["cluster_regions"], mapping="Beryl")
    regions = (
        [[k] for k in np.unique(beryl_reg)] if kwargs["single_region"] else [np.unique(beryl_reg)]
    )
    print(f"Use spikes from brain regions: ", regions[0])
    return regions, beryl_reg


def select_brain_regions(beryl_reg, region):
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask).flatten()
    return reg_clu_ids


def get_spike_data_per_interval(
    times, 
    clusters, 
    interval_begs, 
    interval_ends, 
    interval_len, 
    binsize,
):
    n_intervals = len(interval_begs)

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    cluster_ids = np.unique(clusters)
    n_clusters_in_region = len(cluster_ids)

    binned_spikes = np.zeros((n_intervals, n_clusters_in_region, n_bins))

    intervals = list(zip(np.arange(n_intervals), interval_begs, interval_ends))

    for interval_idx, t_beg, t_end in tqdm(intervals):
        idxs_t = (times >= t_beg) & (times < t_end)
        times_curr = times[idxs_t]
        clust_curr = clusters[idxs_t]

        if times_curr.shape[0] == 0:
            # no spikes in this interval
            binned_spikes_tmp = np.zeros((n_clusters_in_region, n_bins))

            idxs_tmp = np.arange(n_clusters_in_region)

        else:
            # bin spikes
            binned_spikes_tmp, _, cluster_idxs = bincount2D(
                times_curr, clust_curr, xbin=binsize, xlim=[t_beg, t_end]
            )

            # map cluster indices
            _, idxs_tmp, _ = np.intersect1d(
                cluster_ids, cluster_idxs, return_indices=True
            )

        # accumulate
        binned_spikes[interval_idx, idxs_tmp, :] += binned_spikes_tmp[:, :n_bins]

    return binned_spikes


def bin_spiking_data(
    reg_clu_ids, 
    neural_df, 
    intervals=None, 
    trials_df=None, 
    **kwargs
):
    if trials_df is not None:
        intervals = np.vstack([
            trials_df[kwargs["align_time"]] + kwargs["time_window"][0],
            trials_df[kwargs["align_time"]] + kwargs["time_window"][1]
        ]).T
        chunk_len = kwargs["time_window"][1] - kwargs["time_window"][0]
        interval_len = (
            kwargs["time_window"][1] - kwargs["time_window"][0]
        )
    else:
        assert intervals is not None, \
            "Require intervals to segment the recording into chunks including trials and non-trials."
        chunk_len = intervals[0,1] - intervals[0,0]
        interval_len = (
            intervals[0,1] - intervals[0,0]
        )
    # subselect spikes for this region
    spikemask = np.isin(neural_df["spike_clusters"], reg_clu_ids)
    regspikes = neural_df["spike_times"][spikemask]
    regclu = neural_df["spike_clusters"][spikemask]
    clusters_used_in_bins = np.unique(regclu)
    binsize = kwargs.get("binsize", chunk_len)
    
    if chunk_len / binsize == 1.0:
        # one vector of neural activity per interval
        binned, _ = get_spike_counts_in_bins(regspikes, regclu, intervals)
        binned = binned.T  # binned is a 2D array
        binned_list = [x[None, :] for x in binned]
    else:
        binned_array = get_spike_data_per_interval(
            regspikes, regclu,
            interval_begs=intervals[:, 0],
            interval_ends=intervals[:, 1],
            interval_len=interval_len,
            binsize=kwargs["binsize"],
        )
        binned_list = [x.T for x in binned_array]   
    return np.array(binned_list), clusters_used_in_bins


def bin_behaviors(
    eid, 
    behaviors,
    trials_df=None, 
    mask=None, 
    allow_nans=True, 
    **kwargs
):
    if trials_df is None:
        raise ValueError("Visual alignment requires trial IDs and session event timestamps")
    if mask is not None:
        trials_df = trials_df.loc[mask]
    ids = validate_ids(trials_df["original_trial_id"].to_numpy()
                       if "original_trial_id" in trials_df else trials_df.index.to_numpy())
    origins = trials_df[kwargs["align_time"]].to_numpy(dtype=float)
    start, stop = kwargs["time_window"]
    binsize = kwargs["binsize"]
    if not np.isfinite([start, stop, binsize]).all() or stop <= start or binsize <= 0:
        raise ValueError("Invalid alignment window or bin size")
    count = (stop-start)/binsize
    if not np.isclose(count, round(count)):
        raise ValueError("Alignment window must contain a whole number of bins")
    centers = start + (np.arange(round(count)) + .5) * binsize
    behave_dict, mask_dict = {}, {}
    for beh in behaviors:
        if beh != "vision-clip":
            raise ValueError(f"Unsupported visual modality: {beh}")
        archive = load_visual_stimulus(eid, beh)
        values = np.zeros((len(ids), len(centers), 768), dtype=np.float32)
        validity = np.zeros((len(ids), len(centers)), dtype=bool)
        if not archive["skip"]:
            lookup = {int(tid): i for i, tid in enumerate(archive["trial_ids"])}
            for row, tid in enumerate(ids):
                index = lookup.get(int(tid))
                if index is None or not np.isfinite(origins[row]):
                    continue
                values[row], validity[row] = resample_features(
                    archive["times"][index], archive["values"][index],
                    archive["valid"][index], origins[row] + centers)
        if not allow_nans and not validity.all():
            raise ValueError("Incomplete visual coverage; retain validity masks with allow_nans=True")
        behave_dict[beh], mask_dict[beh] = values, validity
    return behave_dict, mask_dict


def prepare_data(access, eid):
    probes = access.probes(eid)
    details = access.metadata(eid).reported
    if not probes:
        raise SkipSession(f"No probe insertions for EID {eid}")
    print(f"Merge {len(probes)} probes for session EID: {eid}")

    clusters_list = []
    spikes_list = []
    source_records = []
    for probe in probes:
        pid, probe_name = probe["id"], probe["name"]
        tmp_spikes, tmp_clusters, sampling_freq = load_spiking_data(
            access, pid, eid=eid, pname=probe_name
        )
        if tmp_spikes is None:
            return None, None, None, None, None
        if tmp_clusters.empty:
            raise SkipSession(f"No clusters for EID {eid}, PID {pid}, probe {probe_name}")
        tmp_clusters["pid"] = pid
        tmp_clusters["probe"] = probe_name
        source_records.append(tmp_clusters.attrs["source"])
        spikes_list.append(tmp_spikes)
        clusters_list.append(tmp_clusters)
    spikes, clusters = merge_probes(spikes_list, clusters_list)

    trial_source = access.load_trials(eid)
    trials_df, trials_mask = load_trials_and_mask(
        access=access, eid=eid, min_rt=0., max_rt=10., trials=trial_source.data,
    )
    good_trials_mask = trials_mask.copy()
        
    trials_df["original_trial_id"] = np.arange(len(trials_df), dtype=np.int64)
    behave_dict = {}  # Visual data are loaded and validated once during binning.
    
    neural_dict = {
        "spike_times": spikes["times"],
        "spike_clusters": spikes["clusters"],
        "cluster_regions": clusters["acronym"].to_numpy(),
    }
        
    meta_data = {
        "eid": eid,
        "subject": details["subject"],
        "lab": details["lab"],
        "sampling_freq": sampling_freq,
        "cluster_channels": list(clusters["channels"]),
        "cluster_regions": list(clusters["acronym"]),
        "good_clusters": list((clusters["label"] >= 1).astype(int)),
        "cluster_depths": list(clusters["depths"]),
        "uuids":  list(clusters["uuids"]),
        "cluster_pids": list(clusters["pid"]),
        "cluster_probes": list(clusters["probe"]),
        "source_datasets": dict(spikes=source_records,
                                trials=json.loads(json.dumps([asdict(s) for s in trial_source.sources], default=str))),
    }

    trials_data = {
        "trials_df": trials_df,
        "trials_mask": trials_mask
    }
    return neural_dict, behave_dict, meta_data, trials_data, good_trials_mask


def standardize_lfp_data(lfp_data, means=None, stds=None):
    K, T, N = lfp_data.shape
    if (means is None) and (stds == None):
        means, stds = np.empty((T, N)), np.empty((T, N))

    std_lfp_data = lfp_data.reshape((K, -1))
    std_lfp_data[np.isnan(std_lfp_data)] = 0
    for t in range(T):
        mean = np.mean(std_lfp_data[:, t*N:(t+1)*N])
        std = np.std(std_lfp_data[:, t*N:(t+1)*N])
        std_lfp_data[:, t*N:(t+1)*N] -= mean
        if std != 0:
            std_lfp_data[:, t*N:(t+1)*N] /= std
        means[t], stds[t] = mean, std
    std_lfp_data = std_lfp_data.reshape(K, T, N)
    return std_lfp_data, means, stds


def align_data(
    binned_spikes, 
    binned_behaviors, 
    binned_lfp=None,
    beh_names=[
        "vision-clip",
    ], 
    trials_mask=None,
    behavior_masks=None,
    trial_index=None,
):
    num_trials = len(binned_spikes)
    if not beh_names:
        raise ValueError("No visual modalities supplied for alignment")
    target_mask = np.isfinite(binned_spikes).all(axis=(1, 2))
    prepared, validity = {}, {}
    for beh in beh_names:
        if beh not in binned_behaviors or len(binned_behaviors[beh]) != num_trials:
            raise ValueError(f"Missing or mismatched modality: {beh}")
        rows = binned_behaviors[beh]
        shape = (binned_spikes.shape[1], 768)
        values = np.zeros((num_trials, *shape), dtype=np.float32)
        valid = np.zeros((num_trials, shape[0]), dtype=bool)
        for row, value in enumerate(rows):
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float32)
            if value.shape != shape:
                raise ValueError(f"Invalid {beh} shape in trial row {row}: {value.shape}")
            valid[row] = np.isfinite(value).all(axis=-1)
            values[row] = np.where(valid[row, :, None], value, 0)
        if behavior_masks is not None:
            supplied = np.asarray(behavior_masks[beh])
            if supplied.shape != valid.shape or supplied.dtype.kind != "b":
                raise ValueError(f"Invalid time validity mask for {beh}")
            valid &= supplied
        values[~valid] = 0
        prepared[beh], validity[beh] = values, valid
        target_mask &= valid.any(axis=1)
    if trials_mask is not None:
        if (trial_index is not None and hasattr(trials_mask, "index")
                and not np.array_equal(np.asarray(trials_mask.index), np.asarray(trial_index))):
            raise ValueError("Trial validity mask index does not match neural trial row order")
        supplied = np.asarray(trials_mask)
        if supplied.shape != (num_trials,) or not np.isin(supplied, [0, 1]).all():
            raise ValueError("Trial validity mask must contain one boolean/0/1 per original row")
        target_mask &= supplied.astype(bool)
    if binned_lfp is not None:
        if len(binned_lfp) != num_trials:
            raise ValueError("LFP trial count mismatch")
        target_mask &= np.isfinite(binned_lfp).all(axis=(1, 2))
    if not target_mask.any():
        raise ValueError("No valid trials with visual coverage remain")
    aligned = {}
    for beh in beh_names:
        aligned[beh] = prepared[beh][target_mask]
        aligned[beh + "_valid"] = validity[beh][target_mask]
    lfp = (standardize_lfp_data(binned_lfp[target_mask])[0]
           if binned_lfp is not None else None)
    return binned_spikes[target_mask], aligned, lfp, target_mask, np.flatnonzero(~target_mask)


def load_visual_stimulus(
    eid,
    target="vision-clip"
):
    """Missing files skip the session; malformed files fail explicitly."""
    if target != "vision-clip":
        raise ValueError(f"Unsupported visual modality: {target}")
    path = get_visual_dir() / f"{eid}_visual_clip.npz"
    try:
        return load_archive(path, eid)
    except FileNotFoundError:
        return dict(trial_ids=[], times=[], values=[], valid=[], skip=True)
