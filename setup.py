from setuptools import setup, find_packages

setup(
    name="jcm",
    version="0.1",
    packages=find_packages(),
    package_dir={"jcm": "jcm"},
    install_requires=[
        "wandb",
        "clean-fid",
        "torchvision",
        "torch",
        "tensorflow",
        "tensorboard",
        "absl-py",
        "flax",
        "jax==0.4.10",
        "dm-haiku",
        "optax",
        "diffrax",
        "ml-collections",
        "requests",
        "scikit-image",
        "termcolor",
        "mpi4py",
        "smart-open[all]",
        "azure-identity",
        "azure-storage-blob",
        "pandas",
        "seaborn",
        "tqdm",
        "huggingface_hub",
        "h5py",
        "flaxmodels",
        "torch-fidelity",
    ],
)
