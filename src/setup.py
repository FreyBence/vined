from setuptools import setup, find_packages

setup(
    name='vined',
    version='0.0.1',
    python_requires='>=3.10,<3.11',
    description='Visual-Neural Encoding and Decoding',
    url='https://github.com/FreyBence/vined',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'timm'
    ],
)
