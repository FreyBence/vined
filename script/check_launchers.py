"""Check launcher interpreter selection and argument boundaries without running jobs."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bash', default='bash')
    args = parser.parse_args()
    for script in (ROOT / 'script').glob('*.sh'):
        subprocess.run([args.bash, '-n', script.as_posix()], check=True)
    with tempfile.TemporaryDirectory(prefix='vined launcher ') as tmp:
        tmp = Path(tmp)
        bin_dir = tmp / 'venv with spaces' / 'bin'
        bin_dir.mkdir(parents=True)
        log = tmp / 'args.log'
        python = bin_dir / 'python'
        python.write_text(
            '#!/usr/bin/env bash\n'
            'printf "%s\\0" "$PWD" "$0" "$@" > "$LAUNCHER_LOG"\n'
            'printf "%s\\0" "$VINED_REPO_ROOT/src" "$PYTHONPATH" > "$LAUNCHER_LOG.pythonpath"\n',
            newline='\n')
        python.chmod(0o755)
        for name, content in {
            'scontrol': '#!/usr/bin/env bash\nprintf "node1\\nnode2\\n"\n',
            'srun': '#!/usr/bin/env bash\nif [[ "$*" == *hostname* ]]; then echo 127.0.0.1; else "$@"; fi\n',
        }.items():
            path = bin_dir / name
            path.write_text(content, newline='\n')
            path.chmod(0o755)
        env = dict(os.environ, VENV_DIR=bin_dir.parent.as_posix(), LAUNCHER_LOG=log.as_posix(),
                   VINED_DATA_DIR=(tmp/'data with spaces').as_posix(),
                   VINED_VISUAL_DIR=(tmp/'visual with spaces').as_posix(),
                   VINED_OUTPUT_DIR=(tmp/'output with spaces').as_posix())
        env['PYTHONPATH'] = (tmp/'extra modules').as_posix()
        def check_pythonpath():
            source, exported, _ = Path(str(log) + '.pythonpath').read_bytes().decode().split('\0')
            assert exported == source + ':' + env['PYTHONPATH'], exported
        env.pop('SLURM_JOB_ID', None)
        env.pop('VINED_REPO_ROOT', None)
        cases = {
            'create_dataset.sh': ['1', 'test-eid'],
            'prepare_data.sh': ['1', 'test-eid'],
            'prepare_visual_stim.sh': ['1', 'test-eid'],
            'train.sh': ['1', 'test-eid', 'train', 'mm', '0', '0.1', 'False', 'random'],
            'eval.sh': ['1', 'test-eid', 'train', 'mm', '0.1', 'random', 'False'],
        }
        for cwd in [ROOT, ROOT/'script']:
            for script, argv in cases.items():
                subprocess.run([args.bash, (ROOT/'script'/script).as_posix(), *argv],
                               cwd=cwd, env=env, check=True, capture_output=True)
                recorded = log.read_bytes().decode().split('\0')
                assert 'test-eid' in recorded, (script, recorded)
                assert 'venv with spaces/bin/python' in recorded[1], recorded
                assert any(' with spaces' in arg for arg in recorded[2:]), recorded
                check_pythonpath()
        # Simulate an sbatch-spooled script and verify worker argv without Slurm.
        spooled = tmp/'spooled.sh'
        spooled.write_bytes((ROOT/'script/train_multi_gpu.sh').read_bytes())
        env.update(SLURM_JOB_ID='123', SLURM_NNODES='2', SLURM_JOB_NODELIST='node[1-2]',
                   SLURM_SUBMIT_DIR=ROOT.as_posix())
        subprocess.run([args.bash, spooled.as_posix(), '1', 'test-eid', 'mm', '0', '0.1', 'random'],
                       cwd=tmp, env=env, check=True, capture_output=True)
        recorded=log.read_bytes().decode().split('\0')
        assert recorded[2:4] == ['-m', 'torch.distributed.run'], recorded
        assert env['VINED_DATA_DIR'] in recorded and env['VINED_OUTPUT_DIR'] in recorded, recorded
        assert '--mixed_training' in recorded, recorded
        check_pythonpath()
        print('PASS: Bash syntax, root/script invocation, venv/path spaces, checkout PYTHONPATH, spooled Slurm worker argv')


if __name__ == '__main__':
    main()
