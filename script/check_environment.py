"""Offline environment checks. Run with the project's venv Python from any cwd."""
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--clip', action='store_true', help='Check cached CLIP weights without downloading')
    parser.add_argument('--session-cache', type=Path, help='Directory of trusted prepared trial .npy files; check two trials')
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 10), sys.version
    assert sys.prefix != sys.base_prefix, 'Use a virtual environment'
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    for name in ['cv2', 'PIL', 'yaml', 'numpy', 'pandas', 'scipy', 'sklearn',
                 'torchvision', 'timm', 'torcheval.metrics', 'transformers',
                 'accelerate', 'ray.tune', 'wandb', 'one.api', 'iblatlas',
                 'utils.ibl_data_utils', 'utils.preprocess_lfp',
                 'loader.make_loader', 'multi_modal.mm', 'trainer.make']:
        importlib.import_module(name)
        print('IMPORT OK:', name, flush=True)
    import cv2
    import numpy as np
    import torch
    from datasets import Dataset, Features, Array2D, load_from_disk
    from utils.paths import REPO_ROOT, visual_dir
    from multi_modal.mm_utils import Attention

    with tempfile.TemporaryDirectory() as tmp:
        video = str(Path(tmp) / 'tiny.mp4')
        writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'mp4v'), 5, (32, 32))
        assert writer.isOpened(), 'MP4 encoder unavailable'
        for _ in range(3):
            writer.write(np.full((32, 32, 3), 127, np.uint8))
        writer.release()
        reader = cv2.VideoCapture(video)
        count = 0
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            assert frame.shape == (32, 32, 3)
            count += 1
        reader.release()
        assert count == 3, count
        values = np.random.default_rng(42).normal(size=(2, 4, 768)).astype('float32')
        dataset = Dataset.from_dict({'vision': values}, features=Features({'vision': Array2D((4, 768), 'float32')}))
        path = str(Path(tmp) / 'dataset')
        dataset.save_to_disk(path)
        restored = load_from_disk(path).with_format('numpy')['vision']
        np.testing.assert_allclose(restored, values)
        old = os.environ.get('VINED_VISUAL_DIR')
        try:
            os.environ['VINED_VISUAL_DIR'] = 'tmp/custom-vision'
            assert visual_dir() == REPO_ROOT / 'tmp/custom-vision'
        finally:
            if old is None:
                os.environ.pop('VINED_VISUAL_DIR', None)
            else:
                os.environ['VINED_VISUAL_DIR'] = old
    print('MP4, dataset and path round trips OK', flush=True)

    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA wheel/driver/device unavailable'
        assert torch.version.cuda == '11.8', torch.version.cuda
    # Exercise the project's actual attention module, including PyTorch SDPA.
    model = Attention(0, 32, 4, True, 0.0, use_rope=True, max_F=4, n_mod=2).to(args.device)
    x = torch.randn(2, 8, 32, device=args.device, requires_grad=True)
    timestamps = torch.arange(8, device=args.device).expand(2, -1)
    y = model(x, timestamp=timestamps)
    loss = y.square().mean()
    loss.backward()
    assert torch.isfinite(loss) and x.grad is not None and torch.isfinite(x.grad).all()
    assert any(p.grad is not None for p in model.parameters())
    print(f'Project attention forward/backward OK: {args.device}; torch={torch.__version__}', flush=True)
    from multi_modal.encoder_embeddings import EncoderEmbedding, INCLUDE_EIDS
    from multi_modal.mm import MultiModal
    from utils.config_utils import update_config
    trials = None
    eid, time_length, neurons = INCLUDE_EIDS[0], 4, 5
    if args.session_cache:
        files = sorted(args.session_cache.glob('*.npy'))
        assert len(files) >= 2, 'Need two prepared trials'
        trials = [np.load(path, allow_pickle=True).item() for path in files[:2]]
        eid = trials[0]['eid']
        assert all(trial['eid'] == eid for trial in trials), 'Use two trials from the same session'
        assert eid in INCLUDE_EIDS, 'Session must be registered in the session lists'
        time_length, neurons = trials[0]['spikes_data'].shape
    config = update_config(str(ROOT / 'src/configs/multi_modal/mm.yaml'), {
        'encoder': {'embedder': {'max_F': time_length, 'n_modality': 2},
                    'transformer': {'hidden_size': 32, 'inter_size': 64, 'n_heads': 4, 'n_layers': 1}}
    })
    included = update_config({'model': 'include:' + str(ROOT / 'src/configs/multi_modal/mm.yaml')})
    assert included.model.model_class == 'MultiModal'
    eids = {eid: neurons}
    embeddings = {mod: EncoderEmbedding(32, 32, config.encoder, stitching=True,
                                        eid_list=eids, mod=mod, max_F=time_length)
                  for mod in ['spike', 'vision-clip']}
    model = MultiModal(embeddings, ['spike', 'vision-clip'], ['vision-clip'], 'mm',
                       config, eid_list=eids).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    def batch():
        result = {}
        for idx, (mod, width) in enumerate([('spike', neurons), ('vision-clip', 768)]):
            data = torch.rand(2, time_length, width, device=args.device)
            if trials:
                key = 'spikes_data' if mod == 'spike' else mod
                data = torch.as_tensor(np.stack([trial[key] for trial in trials]), dtype=torch.float32, device=args.device)
            attn_mask = torch.ones(2, time_length, dtype=torch.int64, device=args.device)
            if trials:
                attn_mask = torch.as_tensor(np.stack([trial['time_attn_mask'] for trial in trials]), dtype=torch.int64, device=args.device)
            result[mod] = dict(inputs=data, targets=data.clone(), eid=[eid, eid],
                inputs_modality=torch.tensor(idx, device=args.device),
                inputs_timestamp=torch.arange(time_length, device=args.device).expand(2, -1),
                inputs_attn_mask=attn_mask,
                eval_mask=torch.ones_like(data), training_mode='random_token')
        return result
    output = model(batch())
    assert torch.isfinite(output.loss), output.loss
    output.loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(batch())
    assert output.mod_preds['spike'].shape == (2, time_length, neurons)
    assert output.mod_preds['vision-clip'].shape == (2, time_length, 768)
    source = f'prepared session {eid}' if trials else 'synthetic data'
    print(f'Multimodal optimizer step and evaluation OK: {source}', flush=True)
    if args.clip:
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
        from prepare_visual_stim import extract_clip_features
        clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14', local_files_only=True).to(args.device).eval()
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', local_files_only=True)
        features = extract_clip_features([Image.new('RGB', (224, 224), 'gray')], clip, processor, args.device, 1)
        assert features.shape == (1, 768) and np.isfinite(features).all()
        print('Cached CLIP feature extraction OK', flush=True)
    for entry in ['train.py', 'finetune.py', 'eval.py', 'prepare_data.py',
                  'create_dataset.py', 'prepare_visual_stim.py']:
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'src' / entry), '--help'],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f'{entry}: {result.stderr}'
        print('CLI OK:', entry, flush=True)


if __name__ == '__main__':
    main()
