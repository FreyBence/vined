import random

import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler

from loader.base import (
    BaseDataset,
    LengthGroupedSampler,
    LengthStitchGroupedSampler,
    SessionSampler,
    WeightedSessionSampler,
)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def calculate_weights(labels):
    unique_classes = np.unique(labels)
    class_counts = np.zeros(len(unique_classes))
    for i, c in enumerate(unique_classes):
        class_counts[i] = (np.array(labels) == c).sum()
    class_weights = 1.0 / class_counts
    weights = np.array([class_weights[int(l)] for l in labels])
    return weights

def make_loader(
    dataset,
    batch_size,
    target=None,
    pad_to_right=True,
    sort_by_depth=False,
    sort_by_region=False,
    pad_value=0.,
    max_time_length=5000,
    max_space_length=100,
    bin_size=0.05,
    brain_region='all',
    load_meta=False,
    dataset_name="ibl",
    stitching=False,
    seed=42,
    shuffle=True,
    weighted_sampler=False,
    data_dir=None,
    mode='train',
    eids=None,
):

    dataset = BaseDataset(
        dataset=dataset,
        target=target,
        pad_value=pad_value,
        max_time_length=max_time_length,
        max_space_length=max_space_length,
        bin_size=bin_size,
        pad_to_right=pad_to_right,
        dataset_name=dataset_name,
        sort_by_depth=sort_by_depth,
        sort_by_region=sort_by_region,
        brain_region=brain_region,
        load_meta=load_meta,
        stitching=stitching,
        data_dir=data_dir,
        mode=mode,
        eids=eids,
    )

    print(f"[DEBUG] BaseDataset length: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError(
            f"BaseDataset is empty in mode={mode}. "
            f"Check filtering, eids, and BaseDataset construction."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    # --------------------------------------------------
    # SIMPLIFIED SINGLE-SESSION LOADER
    # --------------------------------------------------

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if mode == "train" else False,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader
