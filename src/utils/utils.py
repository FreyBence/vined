import glob
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score as r2_score_sklearn
from torcheval.metrics import R2Score
from tqdm import tqdm

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(f"{PROJ_DIR}/data/test_eids.txt") as file:
    test_eids = file.read().splitlines()
    
def dummy_load(stop_event, dummy_size=60000, check_interval=1, device="cuda"):
    dummy_linear = torch.nn.Linear(dummy_size, dummy_size).to(device)
    with torch.no_grad():
        while not stop_event.is_set():
            x_ = dummy_linear(torch.rand(2048,dummy_size).to(device))
            time.sleep(check_interval)  # Adjust the sleep interval as needed

def set_seed(seed):
    print("Global seed set to {}.".format(seed))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def move_batch_to_device(batch, device):
    # if batch values are tensors, move them to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch

def plot_gt_pred(gt, pred, epoch=0,modality="behavior"):
    # plot Ground Truth and Prediction in the same figur
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Ground Truth")
    im1 = ax1.imshow(gt, aspect='auto', cmap='binary')
    
    ax2.set_title("Prediction")
    im2 = ax2.imshow(pred, aspect='auto', cmap='binary')
    
    # add colorbar
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)

    fig.suptitle("Epoch: {}, Mod: {}".format(epoch, 
                                             modality))
    return fig

def plot_neurons_r2(gt, pred, epoch=0, neuron_idx=[],modality="behavior"):
    # Create one figure and axis for all plots
    fig, axes = plt.subplots(len(neuron_idx), 1, figsize=(12, 5 * len(neuron_idx)))
    r2_values = []  # To store R2 values for each neuron
    
    for idx, neuron in enumerate(neuron_idx):
        r2 = r2_score(y_true=gt[:, neuron], y_pred=pred[:, neuron])
        r2_values.append(r2)
        ax = axes if len(neuron_idx) == 1 else axes[idx]
        ax.plot(gt[:, neuron].cpu().numpy(), label="Ground Truth", color="blue")
        ax.plot(pred[:, neuron].cpu().numpy(), label="Prediction", color="red")
        ax.set_title("Neuron: {}, R2: {:.4f}".format(neuron, r2))
        ax.legend()
        # x label
        ax.set_xlabel("Time")
        # y label
        ax.set_ylabel("Rate")
    fig.suptitle("Epoch: {}, Mod: {}, Avg R2: {:.4f}".format(epoch, 
                                                            modality, 
                                                            np.mean(r2_values)))
    return fig

def plt_condition_avg_r2(gt, pred, epoch=0, neuron_idx=0, condition_idx=0, first_n=8, device="cpu"):
    _, unique, counts = np.unique(gt.cpu().numpy(), axis=0, return_inverse=True, return_counts=True)
    trial_idx = (unique == condition_idx)

    if trial_idx.sum() < first_n:
        first_n = trial_idx.sum()

    gt_condition = gt[trial_idx][0,:,neuron_idx]
    pred_condition = pred[trial_idx][:first_n,:,neuron_idx]
    
    # plot line of gt and pred in different colors
    r2 = r2_score(y_true=gt_condition, y_pred=torch.mean(pred_condition, axis=0), device=device)
    gt_condition = gt_condition.cpu().numpy()
    pred_condition = pred_condition.cpu().numpy()
    fig, ax = plt.subplots()
    ax.plot(gt_condition, label="Ground Truth", color="blue")
    # plot all pred trials, and show the range of the first_n trials through the shaded area
    ax.plot(np.mean(pred_condition, axis=0), label="Prediction", color="red")
    ax.fill_between(np.arange(pred_condition.shape[1]), np.min(pred_condition, axis=0), np.max(pred_condition, axis=0), color="red", alpha=0.2)

    ax.set_title("R2: {:.4f}".format(r2))
    ax.legend()
    # x label
    ax.set_xlabel("Time")
    # y label
    ax.set_ylabel("Rate")
    fig.suptitle("Epoch: {}, Neuron: {}, Condition: {}, Avg {} trials".format(epoch, neuron_idx, condition_idx, first_n))
    return fig


from scipy.special import gammaln


def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
            spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            logger.warning(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = 1e-9

    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)


def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)

r2_metric = R2Score()
def r2_score(y_true, y_pred, device="cpu"):
    r2_metric.reset()
    r2_metric.to(device)
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    r2_metric.update(y_pred, y_true)
    return r2_metric.compute().item()

# metrics list, return different metrics results
def metrics_list(gt, pred, metrics=["bps", "r2", "rsquared", "mse", "mae", "acc"], device="cpu"):
    results = {}

    if "bps" in metrics:
        gt, pred = gt.transpose(-1,0).cpu().numpy(), pred.transpose(-1,0).cpu().numpy()
        bps_list = []
        for i in range(gt.shape[-1]): 
            bps = bits_per_spike(pred[:,:,[i]], gt[:,:,[i]])
            if np.isinf(bps):
                bps = np.nan
            bps_list.append(bps)
        mean_bps = np.nanmean(bps_list)
        results["bps"] = mean_bps
    
    if "r2" in metrics:
        r2_list = []
        for i in range(gt.shape[0]):
            r2s = [r2_score(y_true=gt[i].T[k], y_pred=pred[i].T[k], device=device) for k in range(len(gt[i].T))]
            r2_list.append(np.ma.masked_invalid(r2s).mean())
        r2 = np.mean(r2_list)
        results["r2"] = r2
        
    if "behave_r2" in metrics:
        r2_list = []
        gt, pred = gt.transpose(-1,0).cpu().numpy(), pred.transpose(-1,0).cpu().numpy()
        for i in range(gt.shape[0]):
           r2 = r2_score_sklearn(gt[i].flatten(), pred[i].flatten())
           r2_list.append(r2)
        mean_r2 = np.nanmean(r2_list)
        results["behave_r2"] = mean_r2
        
    if "rsquared" in metrics:
        r2 = 0
        for i in range(gt.shape[-1]):
            r2_list = []
            for j in range(gt.shape[0]):
                r2 = r2_score(y_true=gt[j,:,i], y_pred=pred[j,:,i], device=device) 
                r2_list.append(r2)
            r2 += np.mean(r2_list)
        results["rsquared"] = r2 / gt.shape[-1]
        
    if "mse" in metrics:
        mse = torch.mean((gt - pred) ** 2)
        results["mse"] = mse
        
    if "mae" in metrics:
        mae = torch.mean(torch.abs(gt - pred))
        results["mae"] = mae
        
    if "acc" in metrics:
        acc = accuracy_score(gt.cpu().numpy(), pred.cpu().detach().numpy())
        results["acc"] = acc
        
    return results


# plot the GT rate and the Prediction rate (after exp). TODO: add the spikes. unavailable now.
def plot_rate_and_spike(gt, pred, epoch=0, eps=1e-7):
    # find the most active neurons
    row_sum = np.sum(gt, axis=1)
    idxs_top6 = np.argsort(row_sum)[-6:]
    pred_top6 = pred[idxs_top6, :]
    gt_top6 = gt[idxs_top6, :]

    fig, axes = plt.subplots(2, 3, figsize=(24, 6))

    for i, ax in enumerate(axes.flat):
        ax.set_ylim(-2, 3)
        # normalized TODO: change the var to be more serious
        y1 = (gt_top6[i, :] - np.mean(gt_top6[i, :])) / (np.sqrt(np.var(gt_top6[i, :])) + eps)
        y2 = (pred_top6[i, :] - np.mean(pred_top6[i, :])) / (np.sqrt(np.var(pred_top6[i, :])) + eps)
        ax.plot(y1, 'grey')
        ax.plot(y2, 'blue')
        ax.set_title(f"Neuron #{idxs_top6[i]}")

    fig.suptitle(f"Epoch: {epoch}")
    return fig


# plot the trail average rate and spike counts
def plot_avg_rate_and_spike(output, epoch=0):  # output is the list of batches (n_batch, 2, bs, seq_len, n_neurons)

    pred_avg = np.zeros(output[0][0][0].detach().cpu().numpy().shape)
    gt_avg = np.zeros(output[0][0][0].detach().cpu().numpy().shape)
    n_trials = 0

    for i, batch in enumerate(output):
        pred = batch[0].detach().cpu().numpy()
        gt = batch[1].detach().cpu().numpy()
        pred_avg = pred_avg + np.sum(pred, axis=0)
        gt_avg = gt_avg + np.sum(gt, axis=0)
        n_trials = n_trials + pred.shape[0]
    pred_avg = pred_avg.T / n_trials
    gt_avg = gt_avg.T / n_trials

    # find the most active neurons
    row_sum = np.sum(gt_avg, axis=1)
    idxs_top6 = np.argsort(row_sum)[-6:]
    pred_avg_top6 = pred_avg[idxs_top6, :]
    gt_avg_top6 = gt_avg[idxs_top6, :]

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    for i, ax in enumerate(axes.flat):
        ax.set_ylim(0, 2)
        y1 = pred_avg_top6[i, :]
        y2 = gt_avg_top6[i, :]
        ax.plot(y1, 'blue')
        ax.plot(y2, 'grey')
        ax.set_title(f'Neuron: #{idxs_top6[i]}')

    fig.suptitle(f"Epoch: {epoch},Avg from {n_trials} trials")
    return fig


def _add_baseline(ax, aligned_tbins=[40]):
    for tbin in aligned_tbins:
        ax.axvline(x=tbin-1, c='k', alpha=0.2)
    # ax.axhline(y=0., c='k', alpha=0.2)


def raster_plot(ts_, vmax, vmin, whether_cbar, ylabel, ax,
                cmap='bwr',
                aligned_tbins=[40]):
    N, T = ts_.shape
    im = ax.imshow(ts_, aspect='auto', cmap=cmap, vmax=vmax, vmin=vmin)
    for tbin in aligned_tbins:
        ax.annotate('',
            xy=(tbin-1, N),
            xytext=(tbin-1, N+10),
            ha='center',
            va='center',
            arrowprops={'arrowstyle': '->', 'color': 'r'})
    if whether_cbar:
        cbar = plt.colorbar(im, pad=0.01, shrink=.6)
        cbar.ax.tick_params(rotation=90)
    if not (ylabel is None):
        ax.set_ylabel(f"{ylabel}"+f"\n(#trials={N})")
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.spines[['left','bottom', 'right', 'top']].set_visible(False)
        pass
    else:
        ax.axis('off')


def compute_R2_main(y, y_pred, clip=True):
    """
    :y: (K, T, N) or (K*T, N)
    :y_pred: (K, T, N) or (K*T, N)
    """
    N = y.shape[-1]
    if len(y.shape) > 2:
        y = y.reshape((-1, N))
    if len(y_pred.shape) > 2:
        y_pred = y_pred.reshape((-1, N))
    r2s = np.asarray([r2_score_sklearn(y[:, n].flatten(), y_pred[:, n].flatten()) for n in range(N)])
    if clip:
        return np.clip(r2s, 0., 1.)
    else:
        return r2s
    

def _one_hot(arr, T):
    uni = np.sort(np.unique(arr))
    ret = np.zeros((len(arr), T, len(uni)))
    for i, _uni in enumerate(uni):
        ret[:,:,i] = (arr == _uni)
    return ret

def _std(arr):
    mean = np.mean(arr, axis=0) # (T, N)
    std = np.std(arr, axis=0) # (T, N)
    std = np.clip(std, 1e-8, None) # (T, N) 
    arr = (arr - mean) / std
    return arr, mean, std


def get_npy_files(log_dir, 
                  model_mode='mm',
                  num_sessions=1,
                  use_contrastive=True,
                  mixed_training=True,
                    ):
    """
    Get the npy files for each session
    """
    if model_mode == 'mm':
        model_mode = 'ap-wheel'
    elif model_mode == 'decoding':
        model_mode = 'ap'
    elif model_mode == 'encoding':
        model_mode = 'wheel'
    else:
        raise ValueError("Unknown model_mode")
    model_mode = f"inModal-{model_mode}"
    # get the npy files under sesNum-{num_sessions}/set-eval
    pattern = f"{log_dir}/sesNum-{num_sessions}/**/*{model_mode}*/**/*.npy"
    npy_files_ = glob.glob(pattern, recursive=True)
    npy_files = []
    test_eids_ = [ses[:5] for ses in test_eids]
    for npy_file in npy_files_:
        ses = npy_file.split('ses-')[1].split('/')[0]
        if ses in test_eids_:
            npy_files.append(npy_file)
    
    # filter the npy files
    if use_contrastive:
        npy_files = [f for f in npy_files if 'contrast-True' in f]
    else:
        npy_files = [f for f in npy_files if 'contrast-False' in f]
    if mixed_training:
        npy_files = [f for f in npy_files if 'mixedTraining-True' in f]
    else:
        npy_files = [f for f in npy_files if 'mixedTraining-False' in f]

    # get behav modal, decoding
    behav_modal = [f for f in npy_files if 'modal_behavior' in f]
    # remove bps in decoding
    behav_modal = [f for f in behav_modal if 'bps' not in f]

    spike_modal = [f for f in npy_files if 'modal_spike' in f]
    # remove r2 in spike
    spike_modal = [f for f in spike_modal if 'r2' not in f]

    npy_files = {
        'spike': spike_modal,
        'behavior': behav_modal
    }

    return npy_files


def return_spike_bps(npy_files):
    bps_list = []
    all_dict = {}
    for npy_file in npy_files['spike']:
        ses = npy_file.split('ses-')[1].split('/')[0]
        encoding_data = np.load(npy_file, allow_pickle=True)
        mean_bps = np.nanmean(encoding_data)
        bps_list.append(mean_bps)
        all_dict[ses] = mean_bps
        print(f"session {ses} spike encoding bps: {mean_bps}")
    print("total {} sessions of spike encoding".format(len(bps_list)))
    return bps_list, all_dict
