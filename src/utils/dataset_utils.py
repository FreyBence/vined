# import h5py
import os

import numpy as np
import torch
from scipy.sparse import csr_array


class DATASET_MODES:
    train = "train"
    val = "val"
    test = "test"
    trainval = "trainval"

DATA_COLUMNS = ['spikes_sparse_data', 'spikes_sparse_indices', 'spikes_sparse_indptr', 'spikes_sparse_shape','cluster_depths']

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET_EIDS=f"{PROJ_DIR}/eids/train_eids.txt"
TEST_RE_EIDS=f"{PROJ_DIR}/eids/test_eids.txt"

def get_train_eids():
    with open(TARGET_EIDS) as file:
        include_eids = [line.rstrip().replace("'", "") for line in file]
    return include_eids

def get_test_eids():
    with open(TEST_RE_EIDS) as file:
        include_eids = [line.rstrip().replace("'", "") for line in file]
    return include_eids

def get_sparse_from_binned_spikes(binned_spikes):
    sparse_binned_spikes = [csr_array(binned_spikes[i], dtype=np.ubyte) for i in range(binned_spikes.shape[0])]
    spikes_sparse_data_list = [csr_matrix.data.tolist() for csr_matrix in sparse_binned_spikes] 
    spikes_sparse_indices_list = [csr_matrix.indices.tolist() for csr_matrix in sparse_binned_spikes]
    spikes_sparse_indptr_list = [csr_matrix.indptr.tolist() for csr_matrix in sparse_binned_spikes]
    spikes_sparse_shape_list = [csr_matrix.shape for csr_matrix in sparse_binned_spikes]

    return sparse_binned_spikes, spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list

def get_binned_spikes_from_sparse(spikes_sparse_data_list, spikes_sparse_indices_list, spikes_sparse_indptr_list, spikes_sparse_shape_list):
    sparse_binned_spikes = [csr_array((spikes_sparse_data_list[i], spikes_sparse_indices_list[i], spikes_sparse_indptr_list[i]), shape=spikes_sparse_shape_list[i]) for i in range(len(spikes_sparse_data_list))]
    binned_spikes = np.array([csr_matrix.toarray() for csr_matrix in sparse_binned_spikes])
    return binned_spikes

    


def get_data_from_h5(mode, filepath, config):
    r"""
        returns:
            spikes
            rates (None if not available)
            held out spikes (for cosmoothing, None if not available)
        * Note, rates and held out spikes codepaths conflict
    """

    has_rates = False
    NLB_KEY = 'spikes' # curiously, old code thought NLB data keys came as "train_data_heldin" and not "train_spikes_heldin"
    NLB_KEY_ALT = 'data'

    with h5py.File(filepath, 'r') as h5file:
        h5dict = {key: h5file[key][()] for key in h5file.keys()}
        if f'eval_{NLB_KEY}_heldin' not in h5dict: # double check
            if f'eval_{NLB_KEY_ALT}_heldin' in h5dict:
                NLB_KEY = NLB_KEY_ALT
        if f'eval_{NLB_KEY}_heldin' in h5dict: # NLB data, presumes both heldout neurons and time are available
            get_key = lambda key: h5dict[key].astype(np.float32)
            train_data = get_key(f'train_{NLB_KEY}_heldin')
            train_data_fp = get_key(f'train_{NLB_KEY}_heldin_forward')
            train_data_heldout_fp = get_key(f'train_{NLB_KEY}_heldout_forward')
            train_data_all_fp = np.concatenate([train_data_fp, train_data_heldout_fp], -1)
            valid_data = get_key(f'eval_{NLB_KEY}_heldin')
            train_data_heldout = get_key(f'train_{NLB_KEY}_heldout')
            if f'eval_{NLB_KEY}_heldout' in h5dict:
                valid_data_heldout = get_key(f'eval_{NLB_KEY}_heldout')
            else:
                self.logger.warn('Substituting zero array for heldout neurons. Only done for evaluating models locally, i.e. will disrupt training due to early stopping.')
                valid_data_heldout = np.zeros((valid_data.shape[0], valid_data.shape[1], train_data_heldout.shape[2]), dtype=np.float32)
            if f'eval_{NLB_KEY}_heldin_forward' in h5dict:
                valid_data_fp = get_key(f'eval_{NLB_KEY}_heldin_forward')
                valid_data_heldout_fp = get_key(f'eval_{NLB_KEY}_heldout_forward')
                valid_data_all_fp = np.concatenate([valid_data_fp, valid_data_heldout_fp], -1)
            else:
                self.logger.warn('Substituting zero array for heldout forward neurons. Only done for evaluating models locally, i.e. will disrupt training due to early stopping.')
                valid_data_all_fp = np.zeros(
                    (valid_data.shape[0], train_data_fp.shape[1], valid_data.shape[2] + valid_data_heldout.shape[2]), dtype=np.float32
                )

            # NLB data does not have ground truth rates
            if mode == DATASET_MODES.train:
                return train_data, None, train_data_heldout, train_data_all_fp
            elif mode == DATASET_MODES.val:
                return valid_data, None, valid_data_heldout, valid_data_all_fp
        train_data = h5dict['train_data'].astype(np.float32).squeeze()
        valid_data = h5dict['valid_data'].astype(np.float32).squeeze()
        train_rates = None
        valid_rates = None
        if "train_truth" and "valid_truth" in h5dict: # original LFADS-type datasets
            has_rates = True
            train_rates = h5dict['train_truth'].astype(np.float32)
            valid_rates = h5dict['valid_truth'].astype(np.float32)
            train_rates = train_rates / h5dict['conversion_factor']
            valid_rates = valid_rates / h5dict['conversion_factor']
            if config.data.use_lograte:
                train_rates = torch.log(torch.tensor(train_rates) + config.data.LOG_EPSILON)
                valid_rates = torch.log(torch.tensor(valid_rates) + config.data.LOG_EPSILON)

    if mode == DATASET_MODES.train:
        return train_data, train_rates, None, None
    elif mode == DATASET_MODES.val:
        return valid_data, valid_rates, None, None
    else: # test unsupported
        return None, None, None, None


    #-------------
    # New version
    # return [item.id for item in list(list_datasets(author=user_or_org_name))]
