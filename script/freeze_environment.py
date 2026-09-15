"""Record installed versions as pip constraints; keep Torch wheel selection separate."""
import argparse
from importlib.metadata import distributions
from pathlib import Path
import platform
import re
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert sys.version_info[:2] == (3, 10), 'Record constraints from Python 3.10'
    pins = {}
    for dist in distributions():
        name = re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower()
        if name == 'neds':
            continue
        version = dist.version
        if name in ('torch', 'torchvision'):
            version = version.split('+')[0]
        pins[name] = version
    header = (f'# Resolved on {platform.system()} {platform.machine()}, Python {platform.python_version()}.\n'
              '# Generated with script/freeze_environment.py after installation.\n'
              '# Torch CPU/CUDA variants are selected by requirements-torch-*.txt.\n')
    args.output.write_text(header + ''.join(f'{name}=={pins[name]}\n' for name in sorted(pins)), encoding='utf-8')


if __name__ == '__main__':
    main()
