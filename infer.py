import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset
from pytorch_lightning.plugins import DDPPlugin
import torch
from PIL import Image
from torchvision import transforms
from torch import optim, nn, utils, Tensor
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
model = vae_models[config['model_params']['name']](**config['model_params'])
# experiment = VAEXperiment(model,
#                           config['exp_params'])

# load checkpoint
checkpoint = "./logs/VanillaVAE/version_36/checkpoints/epoch=151-step=1117047.ckpt"
autoencoder = VAEXperiment.load_from_checkpoint(checkpoint, vae_model=model,
                          params=config['exp_params'])

# choose your trained nn.Module
encoder = autoencoder.model.encoder
encoder.eval()

unic = "9274"

# Load four images and perform preprocessing
img1 = Image.open("ttf/total/uni" + unic + "_simfang.png")
img2 = Image.open("ttf/total/uni" + unic + "_FZXingHeiJW-EB.png")
img3 = Image.open("ttf/total/uni" + unic + "_FZZJ-LongYTJW.png")
img4 = Image.open("ttf/total/uni" + unic + "_FZZJ-XTDFJW.png")

img_transforms = transforms.Compose([
    transforms.CenterCrop(148),
    transforms.Resize(config['data_params']['patch_size']),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

img1 = img_transforms(img1)
img2 = img_transforms(img2)
img3 = img_transforms(img3)
img4 = img_transforms(img4)

# Get the latent representations of the four images
with torch.no_grad():
    [mu, log_var] = autoencoder.model.encode(torch.stack((img1, img2, img3, img4)))
    embeddings = mu.clone().detach()

with torch.no_grad():
    corner_latents = [embeddings[0], embeddings[1], embeddings[2], embeddings[3]]

# Linear interpolation to generate new latent representations
interpolation_arr = []
for i in range(11):
    for j in range(11):
        if i == 0 and j == 0:
            inter_latent = corner_latents[0]
        elif i == 0 and j == 10:
            inter_latent = corner_latents[2]
        elif i == 10 and j == 0:
            inter_latent = corner_latents[1]
        elif i == 10 and j == 10:
            inter_latent = corner_latents[3]
        else:
            # Calculate distances to the corners
            distances = [
                (i ** 2 + j ** 2) ** 0.5,  # Top-left corner
                ((10 - i) ** 2 + j ** 2) ** 0.5,  # Top-right corner
                (i ** 2 + (10 - j) ** 2) ** 0.5,  # Bottom-left corner
                ((10 - i) ** 2 + (10 - j) ** 2) ** 0.5  # Bottom-right corner
            ]

            # Calculate inverse distances and normalize them
            inv_distances = [1 / d for d in distances]
            inv_distances_sum = sum(inv_distances)
            weights = [d / inv_distances_sum for d in inv_distances]

            # Interpolation based on the calculated weights
            inter_latent = sum(embed * weight for embed, weight in zip(corner_latents, weights))

        # Reconstruction from the interpolated latent representations
        inter_latent = autoencoder.model.decode(inter_latent.unsqueeze(0)).squeeze(0)
        interpolation_arr.append(inter_latent)

# Convert the new latent representations back to images
interpolated_imgs = torch.stack(interpolation_arr).reshape(121, 1, 64, 64)  # Reshape tensor to images

# Save the interpolated images as a square result image
save_image(
    interpolated_imgs,
    "new_jian.png",
    normalize=True,
    nrow=11,  # Arrange the images in a square format
)