import argparse
import logging
import os
import json
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
from tqdm import tqdm

from loader.base import BaseDataset
from utils.cache_manifest import CACHE_SCHEMA, preprocessing_sources, aligned_hashes, preprocessing_packages
from utils.provenance import file_hash, fingerprint, write_json, package_versions
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import load_ibl_dataset
from utils.utils import set_seed

logging.basicConfig(level=logging.INFO) 

neural_acronyms = {
    "ap": "spike",
}
static_acronyms = {}
dynamic_acronyms = {
    "vision-clip": "vision-clip",
}

model_config = "src/configs/multi_modal/mm.yaml"
kwargs = {"model": f"include:{model_config}"}
config = config_from_kwargs(kwargs)
config = update_config("src/configs/multi_modal/trainer_mm.yaml", config)
set_seed(config.seed)

# ------ 
# SET UP
# ------ 
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="EXAMPLE_EID")
ap.add_argument("--base_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--data_path", type=str, default="EXAMPLE_PATH")
ap.add_argument("--num_sessions", type=int, default=1)
ap.add_argument("--model_mode", type=str, default="mm")
ap.add_argument("--mask_mode", type=str, default="temporal")
ap.add_argument("--mask_ratio", type=float, default=0.1)
ap.add_argument("--mixed_training", action="store_true")
ap.add_argument(
    "--modality", nargs="+", 
    default=["ap", "vision-clip"]
)
args = ap.parse_args()


eid = args.eid
base_path = args.base_path
model_mode = args.model_mode
modality = args.modality
config["model"]["masker"]["mode"] = args.mask_mode
config["model"]["masker"]["ratio"] = args.mask_ratio

logging.info(f"EID: {eid} model mode: {args.model_mode} mask ratio: {args.mask_ratio}")
logging.info(f"Available modality: {modality}")

neural_mods, static_mods, dynamic_mods = [], [], []
for mod in modality:
    if mod in neural_acronyms:
        neural_mods.append(neural_acronyms[mod])
    elif mod in static_acronyms:
        static_mods.append(static_acronyms[mod])   
    elif mod in dynamic_acronyms:
        dynamic_mods.append(dynamic_acronyms[mod])   

if model_mode == "mm":
    input_mods = output_mods = neural_mods + dynamic_mods
elif model_mode == "decoding":
    input_mods = neural_mods
    output_mods = dynamic_mods
elif model_mode == "encoding":
    input_mods = static_mods + dynamic_mods
    output_mods = neural_mods
else:
    raise ValueError(f"Model mode {model_mode} not supported.")

modal_filter = {"input": input_mods, "output": output_mods}


# ---------
# LOAD DATA
# ---------
train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(
    args.data_path,  
    args.data_path,
    num_sessions=args.num_sessions,
    eid = eid if args.num_sessions == 1 else None,
    use_re=True,
    split_method="predefined",
    test_session_eid=[],
    batch_size=1,
    seed=config.seed
)

max_space_length = max(list(meta_data["eid_list"].values()))
logging.info(f"MAX space length to pad spike data to: {max_space_length}")

# Cache generations are immutable. Each session pointer changes only after all
# splits have been written and checked; unreferenced files are never discovered.
options = dict(target=[mod for mod in modality if mod in dynamic_acronyms],
               load_meta=config.data.load_meta, pad_to_right=True, pad_value=-1.,
               max_time_length=config.data.max_time_length, max_space_length=max_space_length,
               dataset_name=config.data.dataset_name, sort_by_depth=config.data.sort_by_depth,
               sort_by_region=config.data.sort_by_region, stitching=True,
               bin_size=0.05, brain_region="all")
root = Path(args.data_path) / ('ibl_mm' if args.num_sessions == 1 else f'ibl_mm_{args.num_sessions}')
root.mkdir(parents=True, exist_ok=True)
generation = root / 'generations' / uuid4().hex
generation.mkdir(parents=True)
manifests = {}
for session_eid in sorted(meta_data["eid_list"]):
    provenance_path = Path(args.data_path) / f"{session_eid}_aligned" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["fingerprint"] != fingerprint({k: v for k, v in provenance.items() if k != "fingerprint"}):
        raise ValueError("Aligned provenance fingerprint is invalid")
    manifests[session_eid] = dict(schema_version=CACHE_SCHEMA, eid=session_eid,
        options=options, sources=preprocessing_sources(),
        preprocessing_packages=preprocessing_packages(),
        aligned_provenance_sha256=file_hash(provenance_path),
        aligned_files=aligned_hashes(provenance_path.parent), packages=package_versions(),
        provenance_id=provenance["fingerprint"], samples=[])

for split, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
    prepared = BaseDataset(dataset, **options)
    for row in tqdm(range(len(prepared)), desc=split):
        original = dataset[row]
        sample = prepared[row]
        session_eid, trial_id = sample["eid"], int(sample["trial_id"])
        manifest = manifests[session_eid]
        if original.get("provenance_id") != manifest["provenance_id"] or original.get("split") != split:
            raise ValueError("Aligned sample provenance/split does not match manifest; rebuild aligned data")
        name = f"{session_eid}_{split}_{trial_id}.npy"
        destination = generation / name
        if destination.exists():
            raise ValueError("Duplicate original trial in cache generation")
        sample["split"] = split
        sample["provenance_id"] = manifest["provenance_id"]
        np.save(destination, sample)
        restored = np.load(destination, allow_pickle=True).item()
        shapes = {key: list(np.asarray(value).shape) for key, value in sample.items()}
        if restored["eid"] != session_eid or int(restored["trial_id"]) != trial_id:
            raise ValueError("Cache output identity verification failed")
        if {key: list(np.asarray(value).shape) for key, value in restored.items()} != shapes:
            raise ValueError("Cache output shape verification failed")
        manifest["samples"].append(dict(file=destination.relative_to(root).as_posix(),
            split=split, trial_id=trial_id, shapes=shapes, sha256=file_hash(destination)))

for session_eid, manifest in manifests.items():
    identities = [r["trial_id"] for r in manifest["samples"]]
    if len(identities) != len(set(identities)) or not identities:
        raise ValueError("Empty or duplicate cache sample identities")
    provenance = json.loads((Path(args.data_path) / f"{session_eid}_aligned" / "provenance.json").read_text(encoding="utf-8"))
    if aligned_hashes(Path(args.data_path) / f"{session_eid}_aligned") != manifest["aligned_files"]:
        raise ValueError("Aligned dataset changed during cache generation")
    for split, ids in provenance["split"]["memberships"].items():
        if set(ids) != {r["trial_id"] for r in manifest["samples"] if r["split"] == split}:
            raise ValueError("Cache generation does not contain every declared split member")
    manifest["fingerprint"] = fingerprint(manifest)
    with tempfile.NamedTemporaryFile(dir=root, suffix=".json", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        write_json(temporary_path, manifest)
        os.replace(temporary_path, root / f"{session_eid}.manifest.json")
    finally:
        temporary_path.unlink(missing_ok=True)
logging.info("Published complete cache manifests")
