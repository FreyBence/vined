import multiprocessing
from utils.paths import visual_dir as get_visual_dir
import os
import sys
import uuid
from functools import partial
from pathlib import Path

import brainbox.behavior.dlc as dlc
import numpy as np
import pandas as pd
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.population.decode import get_spike_counts_in_bins
from iblatlas.regions import BrainRegions
from iblutil.numerical import bincount2D, ismember
from scipy.interpolate import interp1d
from tqdm import *

DYNAMIC_VARS = ["vision-clip"]

def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result


def load_spiking_data(one, pid, compute_metrics=False, qc=None, **kwargs):
    eid = kwargs.pop("eid", "")
    pname = kwargs.pop("pname", "")
    sampling_freq = 30_000
    spike_loader = SpikeSortingLoader(pid=pid, one=one, eid=eid, pname=pname)
    
    spikes, clusters, channels = spike_loader.load_spike_sorting()
    clusters_labeled = SpikeSortingLoader.merge_clusters(
        spikes, clusters, channels, compute_metrics=compute_metrics
    )
    if clusters_labeled is None:
        return None, None, None
    else:
        clusters_labeled = clusters_labeled.to_df()
    
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
        cluster_max = clusters.index.max() + 1
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
    one, 
    eid, 
    min_rt=0.0, 
    max_rt=10., 
    nan_exclude="default", 
    min_trial_len=None,
    max_trial_len=10, 
    exclude_unbiased=False, 
    exclude_nochoice=True, 
    sess_loader=None,
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

    if sess_loader is None:
        sess_loader = SessionLoader(one=one, eid=eid)

    if sess_loader.trials.empty:
        sess_loader.load_trials()

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

    mask = ~sess_loader.trials.eval(query)
    return sess_loader.trials, mask


def list_brain_regions(neural_dict, **kwargs):
    brainreg = BrainRegions()
    beryl_reg = brainreg.acronym2acronym(neural_dict["cluster_regions"], mapping="Beryl")
    regions = (
        [[k] for k in np.unique(beryl_reg)] if kwargs["single_region"] else [np.unique(beryl_reg)]
    )
    print(f"Use spikes from brain regions: ", regions[0])
    return regions, beryl_reg


def select_brain_regions(regressors, beryl_reg, region, **kwargs):
    reg_mask = np.isin(beryl_reg, region)
    reg_clu_ids = np.argwhere(reg_mask).flatten()
    return reg_clu_ids


def create_intervals(start_time, end_time, interval_len):
    interval_begs = np.arange(
        start_time, end_time-interval_len, interval_len
    )
    interval_ends = np.arange(
        start_time+interval_len, end_time, interval_len
    )
    return np.c_[interval_begs, interval_ends]


def get_spike_data_per_interval(
    times, 
    clusters, 
    interval_begs, 
    interval_ends, 
    interval_len, 
    binsize,
    n_workers=None
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

            if np.isnan(t_beg) or np.isnan(t_end):
                t_idxs = np.nan * np.ones(n_bins)
            else:
                t_idxs = np.arange(t_beg, t_end + binsize / 2, binsize)

            idxs_tmp = np.arange(n_clusters_in_region)

        else:
            # bin spikes
            binned_spikes_tmp, t_idxs, cluster_idxs = bincount2D(
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
    n_workers=os.cpu_count(), 
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
            n_workers=n_workers
        )
        binned_list = [x.T for x in binned_array]   
    return np.array(binned_list), clusters_used_in_bins


def get_behavior_per_interval(
    target_times, 
    target_vals, 
    intervals=None, 
    trials_df=None, 
    allow_nans=False,
    n_workers=None,
    **kwargs
):
    binsize = kwargs["binsize"]

    if trials_df is not None:
        align_event = kwargs["align_time"]
        align_interval = kwargs["time_window"]
        interval_len = align_interval[1] - align_interval[0]
        align_times = trials_df[align_event].values
        interval_begs = align_times + align_interval[0]
        interval_ends = align_times + align_interval[1]
    else:
        assert intervals is not None, \
            "Require intervals to segment the recording into chunks including trials and non-trials."
        interval_begs, interval_ends = intervals.T
        interval_len = interval_ends[0] - interval_begs[0]  # fallback if needed

    n_intervals = len(interval_begs)

    if np.all(np.isnan(interval_begs)) or np.all(np.isnan(interval_ends)):
        print("Interval times all nan")
        good_interval = np.nan * np.ones(interval_begs.shape[0])
        return [], [], good_interval, []

    # np.ceil because we want to make sure our bins contain all data
    n_bins = int(np.ceil(interval_len / binsize))

    # split data into intervals
    idxs_beg = np.searchsorted(target_times, interval_begs, side="right")
    idxs_end = np.searchsorted(target_times, interval_ends, side="left")

    target_times_og_list = [target_times[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]
    target_vals_og_list = [target_vals[ib:ie] for ib, ie in zip(idxs_beg, idxs_end)]

    # outputs
    target_times_list = [None] * n_intervals
    target_vals_list = [None] * n_intervals
    good_interval = [None] * n_intervals
    skip_reasons = [None] * n_intervals

    for interval_idx in tqdm(range(n_intervals)):
        target_time = target_times_og_list[interval_idx]
        target_val = target_vals_og_list[interval_idx]

        is_good_interval, x_interp, y_interp = False, None, None

        if len(target_val) == 0:
            skip_reason = "target data not present"

        elif np.sum(np.isnan(target_val)) > 0 and not allow_nans:
            skip_reason = "nans in target data"

        elif np.isnan(interval_begs[interval_idx]) or np.isnan(interval_ends[interval_idx]):
            skip_reason = "bad interval data"

        elif np.abs(interval_begs[interval_idx] - target_time[0]) > binsize:
            skip_reason = "target data starts too late"

        elif np.abs(interval_ends[interval_idx] - target_time[-1]) > binsize:
            skip_reason = "target data ends too early"

        else:
            is_good_interval = True
            skip_reason = None

            x_interp = np.linspace(
                interval_begs[interval_idx] + binsize,
                interval_ends[interval_idx],
                n_bins
            )

            if len(target_val.shape) > 1 and target_val.shape[1] > 1:
                n_dims = target_val.shape[1]
                y_interp_tmps = []
                for n in range(n_dims):
                    y_interp_tmps.append(
                        interp1d(
                            target_time,
                            target_val[:, n],
                            kind="linear",
                            fill_value="extrapolate"
                        )(x_interp)
                    )
                y_interp = np.hstack([y[:, None] for y in y_interp_tmps])
            else:
                y_interp = interp1d(
                    target_time,
                    target_val,
                    kind="linear",
                    fill_value="extrapolate"
                )(x_interp)

        # store results
        good_interval[interval_idx] = is_good_interval
        target_times_list[interval_idx] = x_interp
        target_vals_list[interval_idx] = y_interp
        skip_reasons[interval_idx] = skip_reason

    return target_times_list, target_vals_list, np.array(good_interval), skip_reasons   


def load_anytime_behaviors(one, eid, n_workers=None):

    behaviors = [
        "vision-clip"
    ]

    behave_dict = {}

    for beh in tqdm(behaviors):
        behave_dict[beh] = load_visual_stimulus(eid, beh)

    return behave_dict


def bin_behaviors(
    one, 
    eid, 
    behaviors,
    intervals=None, 
    trials_only=False, 
    trials_df=None, 
    mask=None, 
    allow_nans=True, 
    n_workers=os.cpu_count(),
    **kwargs
):
    behave_dict, mask_dict = {}, {}

    if mask is not None:
        trials_df = trials_df[mask]

    if trials_df is not None:
        n_trials = len(trials_df)
        behave_mask = np.ones(n_trials, dtype=bool)
    else:
        assert intervals is not None, \
            "Require intervals to segment the recording into chunks including trials and non-trials."
        n_trials = len(intervals)
        behave_mask = np.ones(n_trials, dtype=bool)

    for beh in behaviors:
        target_dict = load_visual_stimulus(eid, target="vision-clip")

        if target_dict["skip"] is False:
            trial_ids = np.asarray(target_dict["trial_ids"], dtype=int)
            target_times_list = target_dict["times"]
            target_vals_list = target_dict["values"]

            n_bins = int(np.ceil(
                (kwargs["time_window"][1] - kwargs["time_window"][0]) / kwargs["binsize"]
            ))

            full_vals = [None] * n_trials
            beh_mask = np.zeros(n_trials, dtype=bool)

            for tid, trial_times, trial_vals in zip(trial_ids, target_times_list, target_vals_list):
                tid = int(tid)

                if tid >= n_trials:
                    continue

                if trial_times is None or trial_vals is None:
                    continue

                trial_times = np.asarray(trial_times)
                trial_vals = np.asarray(trial_vals)

                if len(trial_times) < 2:
                    continue

                x_interp = np.linspace(trial_times[0], trial_times[-1], n_bins)

                y_interp_tmps = []
                for n in range(trial_vals.shape[1]):
                    y_interp_tmps.append(
                        interp1d(
                            trial_times,
                            trial_vals[:, n],
                            kind="linear",
                            fill_value="extrapolate"
                        )(x_interp)
                    )

                y_interp = np.stack(y_interp_tmps, axis=-1).astype(np.float32)

                full_vals[tid] = y_interp
                beh_mask[tid] = True

            behave_dict[beh] = np.array(full_vals, dtype=object)
            mask_dict[beh] = beh_mask
            behave_mask = np.logical_and(behave_mask, beh_mask)

    if not allow_nans:
        for k, v in behave_dict.items():
            behave_dict[k] = behave_dict[k][behave_mask]

    return behave_dict, mask_dict


def prepare_data(one, eid, params, n_workers=os.cpu_count()):
    pids, probe_names = one.eid2pid(eid) 
    details = one.get_details(eid)
    print(f"Merge {len(probe_names)} probes for session EID: {eid}")

    clusters_list = []
    spikes_list = []
    for pid, probe_name in zip(pids, probe_names):
        tmp_spikes, tmp_clusters, sampling_freq = load_spiking_data(
            one, pid, eid=eid, pname=probe_name
        )
        if tmp_spikes is None:
            return None, None, None, None, None
        tmp_clusters["pid"] = pid
        spikes_list.append(tmp_spikes)
        clusters_list.append(tmp_clusters)
    spikes, clusters = merge_probes(spikes_list, clusters_list)

    _, good_trials_mask = load_trials_and_mask(one=one, eid=eid)

    trials_df, trials_mask = load_trials_and_mask(
        one=one, eid=eid, min_rt=0., max_rt=10., 
    )
        
    behave_dict = load_anytime_behaviors(one, eid, n_workers=n_workers)
    
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
        # "cluster_qc": {k: np.asarray(v) for k, v in clusters.to_dict("list").items()},
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
    nan_thresh=0.3,
):
    num_trials = len(binned_spikes)
    
    target_mask = np.ones(num_trials, dtype=bool)

    for beh in beh_names:
        beh_mask = np.array([x is not None for x in binned_behaviors[beh]], dtype=bool)
        nan_ratio = 1 - beh_mask.mean()
        print(f"{beh} has {nan_ratio*100:.1f}% NaN trials.")
        if nan_ratio >= nan_thresh:
            print(f"Remove {beh} due to too many NaN trials!")
        else:
            target_mask &= beh_mask

    if trials_mask is not None:
        trials_mask = list(trials_mask.to_numpy().astype(int))
        target_mask = np.logical_and(
            target_mask,
            beh_mask
        )

    bad_trial_idxs = np.argwhere(np.array(target_mask) == 0)

    aligned_binned_spikes = np.delete(binned_spikes, bad_trial_idxs, axis=0)
    if binned_lfp is not None:
        aligned_binned_lfp, means, stds = standardize_lfp_data(
            np.delete(binned_lfp, bad_trial_idxs, axis=0)
        )
    else:
        aligned_binned_lfp = None

    num_trials = len(aligned_binned_spikes)
    aligned_binned_behaviors = {}
    for beh in beh_names:
        beh_trials = np.delete(np.array(binned_behaviors[beh], dtype=object), bad_trial_idxs, axis=0)

        if beh in DYNAMIC_VARS:
            aligned_binned_behaviors[beh] = np.stack(
                [np.asarray(y, dtype=np.float32) for y in beh_trials],
                axis=0
            )
        else:
            aligned_binned_behaviors[beh] = np.array(beh_trials)

    return (
        aligned_binned_spikes, 
        aligned_binned_behaviors,
        aligned_binned_lfp, 
        target_mask, 
        bad_trial_idxs
    )


def load_visual_stimulus(
    eid,
    target="vision-clip"
):
    """
    Load precomputed CLIP visual embeddings.

    Expected file format:
        {visual_dir}/{eid}_visual_clip.npz

    Required arrays:
        - times: [N_frames]
        - features: [N_frames, clip_dim]
    """
    visual_dir = str(get_visual_dir())
    try:
        visual_path = os.path.join(
            visual_dir,
            f"{eid}_visual_clip.npz"
        )

        data = np.load(visual_path, allow_pickle=True)

        beh_dict = {
            "trial_ids": data["trial_ids"],
            "times": data["times"],
            "values": data["features"],
            "skip": False,
        }

    except BaseException as e:
        print(f"Error loading visual stimulus for {eid}")
        print(e)

        beh_dict = {
            "times": None,
            "values": None,
            "skip": True
        }

    return beh_dict
    