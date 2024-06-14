

# Implementation of the original diffusion paper

Paper: https://arxiv.org/abs/2006.11239


## Setup

- Create a conda environment with the following command:

```bash
conda env create -f environment.yml python=3.10
```


- Download the CIFAR10 dataset using the [notebooks/download-and-explore-dataset.ipynb](notebooks/download-and-explore-dataset.ipynb). 

- Set the env in [diffusion/environments/environments.py](diffusion/environments/environments.py), which should look something like this

```text
DATA_DIR="/home/viktor/Documents/diffusion/data" # where you'll store the cifar10 dataset
DATA_DUMPS_DIR = "/opt/diffusion/data_dumps" # where you'll store the processed data
MODEL_OUT_DIR=f"{DATA_DUMPS_DIR}/models" # where you'll store the trained models
```

- 


# Train


To train a CIFAR10 model, run the following command:
```bash
python tools/train.py
```


# Sample images

To sample images from a trained model, run the following command:


To sample CIFAR10 images, run the following command:


```bash
python tools/sample.py \
    "model_path=WrappedUNetModel-bs128-abs128-llr-4.8-emaTrue-2024-02-17-10-39-45-last-model" \
    "dataset_name=cifar10" \
    "n_imgs=32" \
    "batch_size=2" \
    output.save_all=True \
    output.save_last=False \
```



To sample CelebA-HQ images, run the following command:



```bash
python tools/sample.py \
    "model_path=WrappedUNetModel-bs4-abs4-llr-4.5-emaTrue-2024-02-16-10-53-52-last-model" \
    "dataset_name=celeb_hq" \
    "n_imgs=32" \
    "batch_size=2" \
    output.save_all=True \
    output.save_last=False
```