from tqdm import tqdm
import numpy as np
import pandas as pd
from session_data import AccessPolicy
from utils.sessions import SkipSession
from dataclasses import asdict
import json
import scipy
from ibldsp.utils import rms, fcn_cosine
import spikeinterface.preprocessing as si
from spikeinterface.preprocessing import phase_shift

BANDS = {
    'delta': [0, 4], 
    'theta': [4, 10], 
    'alpha': [8, 12], 
    'beta': [15, 30], 
    'gamma': [30, 90], 
    'lfp': [0, 90],
}

# ----------------
# helper functions
# ----------------

def _get_power_in_band(fscale, period, band):
    band = np.array(band)
    fweights = fcn_cosine([-np.diff(band), 0])(-abs(fscale - np.mean(band)))
    p = 10 * np.log10(np.sum(period * fweights / np.sum(fweights), axis=-1))  # dB relative to v/sqrt(Hz)
    return p

def lf(data, fs, bands=None):
    """Computes the LF features from a numpy array.
    
    :param data: numpy array with the data (channels, samples)
    :param fs: sampling interval (Hz)
    :param bands: dictionary with the bands to compute (default: BANDS constant)
    :return: pandas dataframe with the columns ['channel', 'rms_lf', 'psd_delta', 'psd_theta', 'psd_alpha', 'psd_beta',
       'psd_gamma', 'psd_lfp']
    """
    bands = BANDS if bands is None else bands
    nc = data.shape[0]  # number of channels
    fscale, period = scipy.signal.periodogram(data, fs)
    df_chunk = pd.DataFrame()
    df_chunk['channel'] = np.arange(nc)
    df_chunk['rms_lf'] = rms(data, axis=-1)
    for b in BANDS:
        df_chunk[f"psd_{b}"] = _get_power_in_band(fscale, period, bands[b])
    return df_chunk


def split(data, window_length, time_step):
    """
    Split the data into segments using a sliding window approach.
    
    Parameters:
    - data: numpy array of shape (n_trials, n_channels, n_timepoints)
    - window_length: int, length of each window in timepoints
    - time_step: int, step size between windows in timepoints
    
    Returns:
    - numpy array of shape (n_trials, n_channels, n_windows, window_length)
    """
    n_trials, n_channels, n_timepoints = data.shape
    n_windows = (n_timepoints - window_length) // time_step + 1
    
    segments = np.zeros((n_trials, n_channels, n_windows, window_length))
    
    for i in range(n_windows):
        start = i * time_step
        end = start + window_length
        segments[:, :, i, :] = data[:, :, start:end]
    
    return segments

# ------------------
# load & preprocess
# ------------------
def prepare_lfp(
    access, eid, mask=None, fs=2500.0, dead_channel_threshold=0.,
    trials=None, return_sources=False, **kwargs
):
    """Load, preprocess, and merge LFP from both probes.
    
    Args:
        mask: used to filter out trials; need to be consistent with mask for AP and behavior. 
    """
    trial_window = kwargs["time_window"]
    align_time = kwargs["align_time"]
    
    probes = access.probes(eid)
    if not probes:
        raise SkipSession(f"No ephys insertions for EID {eid}")
    if trials is None:
        trials = access.load_trials(eid).data
    stimOn_times = trials[align_time]
    if mask is not None:
        stimOn_times = stimOn_times[mask]
    stimOn_times = np.asarray(stimOn_times, dtype=float)
    if not len(stimOn_times):
        raise SkipSession(f"No LFP trial rows for EID {eid}")

    lfp_per_probe = []
    sources = []
    for probe in probes:
        print(probe["name"])
        source = access.load_ephys(eid, pid=probe["id"], pname=probe["name"], band="lf",
                                   stream=access.policy == AccessPolicy.REMOTE_ALLOWED)
        rec_si_stream = source.recording
        if not np.isclose(rec_si_stream.get_sampling_frequency(), fs):
            raise ValueError(f"LFP sampling frequency differs from requested {fs} Hz for {probe['name']}")
        sources.append(json.loads(json.dumps(asdict(source.source), default=str)))
        rec_phs = phase_shift(rec_si_stream) # channel rephasing

        # Preserve the existing LFP band; installed SpikeInterface requires an
        # explicit low-cutoff opt-in. Its automatic margins cover filter edges.
        rec_bp = si.bandpass_filter(rec_phs, freq_min=0.5, freq_max=250,
                                    margin_ms="auto", ignore_low_freq_error=True)
        bad_chans, _ = si.detect_bad_channels(
            rec_bp, dead_channel_threshold=dead_channel_threshold, num_random_chunks=100, seed=0
        )
    
        lfp_per_trial = []
        for trial_idx in tqdm(range(len(stimOn_times)), total=len(stimOn_times)):
            t_event = stimOn_times[trial_idx]
            # Keep unavailable trial rows aligned with the other modalities.
            if not np.isfinite(t_event):
                lfp_per_trial.append(np.full((rec_bp.get_num_channels(),
                                             int(round((trial_window[1] - trial_window[0]) * fs))), np.nan))
                continue
            sample_lf = int(np.floor(source.times_to_samples(t_event)))
            first, last = (
                int(trial_window[0] * fs) + sample_lf, 
                int(trial_window[1] * fs + sample_lf)
            )
            chunk = rec_bp.frame_slice(start_frame=first, end_frame=last) 
            rec_prec = si.interpolate_bad_channels(chunk, bad_chans)
            rec_prec = si.common_reference(rec_prec, reference='global', operator='median')
            lfp_per_trial.append(rec_prec.get_traces().T)
        
        lfp_per_trial = np.array(lfp_per_trial)
        lfp_per_trial = lfp_per_trial.transpose(0,2,1)
        
        lfp_per_probe.append(lfp_per_trial)
    lfp_per_probe = np.concatenate(lfp_per_probe, axis=2)
    print("Preprocessed LFP data shape: ", lfp_per_probe.shape)
    
    return (lfp_per_probe, sources) if return_sources else lfp_per_probe

# ------------------
# feature extraction
# ------------------
def featurize_lfp(lfp_data, bin_size=200, samp_freq=2500, batch_size=50):
    """Extract Power spectral density for LFP.
    
    Args:
        lfp_data: preprocessed LFP data from prepare_lfp().
        bin_size: millisecond (ms).
        samp_freq: sampling frequency. 
    """
    n_trials, n_time_samples, _ = lfp_data.shape
    assert n_time_samples % bin_size == 0, f"n_time_samples={n_time_samples} has to be a multiple of bin_size."
    lfp_data = np.transpose(lfp_data, (0,2,1))

    ### Use new preprocessing method: 
    window_length = 743  # 300 ms
    overlap = 700
    time_step = window_length - overlap  # 20 ms step size
    
    batch_idxs = list(range(0, n_trials, batch_size)) 
    n_batches = len(batch_idxs)
    all_psd = {}
    for name in ['psd_lfp']:
        print(f'Frequency Band: {name}')
        l_power_bands = []
        for i, batch_idx in enumerate(batch_idxs):
            print(f'{i+1} / {n_batches} Batch ..')
            if i < n_batches - 1:
                segmented_data = split(lfp_data[batch_idx:batch_idxs[i+1]], window_length, time_step)
            else:
                segmented_data = split(lfp_data[batch_idx:], window_length, time_step)
        
            _n_trials = segmented_data.shape[0]
            
            power_bands = []
            for trial_idx in tqdm(range(_n_trials), total=_n_trials):
                tmp = []
                for tbin_idx in range(segmented_data.shape[2]):  
                    tmp.append(
                        np.array(
                            lf(segmented_data[trial_idx,:,tbin_idx,:], fs=samp_freq)[name]
                        )
                    )
                power_bands.append(tmp)
            l_power_bands.append(power_bands)
            del segmented_data

        binned_bands = np.concatenate(l_power_bands, 0)
        del l_power_bands

        all_psd[name] = binned_bands
        del binned_bands
        
    return all_psd

