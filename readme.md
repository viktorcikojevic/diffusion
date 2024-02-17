

# Implementation of the original diffusion paper

Paper: https://arxiv.org/abs/2006.11239


## Installation

Create a conda environment with the following command:

```bash
conda env create -f environment.yml
```

You also need to set up the environment variables in [src/diffusion/environments/environments.py](src/diffusion/environments/environments.py).



Download the Celeb HQ from https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256


# Train


To train a CIFAR10 model, run the following command:
```bash
python tools/train_nn_hydra.py
```

To train a CelebA-HQ model, run the following command:
```bash
python tools/train_nn_hydra.py \
	dataset=celeb_hq \
	model.kwargs.attention_resolutions.0=16 \
	model.kwargs.model_channels=64 \
	batch_size=4 \
	apparent_batch_size=4 \
	group_name="celeb_hq" \
	val_n_imgs=8 \
```