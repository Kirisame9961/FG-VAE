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
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw

def compare_images_pixel(img1_path, img2_path):
    # 读取两张图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 确保两张图片具有相同的尺寸
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # 计算两张图片的差异
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 将差异图像转换为二值图像
    _, threshold = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    # 计算相似度
    similarity = np.mean(threshold)

    return 255 - similarity

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

material = []
for picture in interpolation_arr:
    temp_result = picture.clone().detach().reshape(1, 1, 64, 64)
    material.append(temp_result)

output_dir = 'testpic'
os.makedirs(output_dir, exist_ok=True)
for i, image in enumerate(material):
    image_path = os.path.join(output_dir, f'{i}.png')
    save_image(image.squeeze(), image_path)

def calculate_weights(similarities):
    exp_sims = np.exp(similarities)
    weights = exp_sims / np.sum(exp_sims)
    return weights

color = []
count = 0
while count <= 120:
    if count == 0 : temp_block = Image.new("RGBA", (64,64),(217,83,25,255))
    elif count == 10: temp_block = Image.new("RGBA", (64,64), (237,177,32,255))
    elif count == 110: temp_block = Image.new("RGBA",(64,64),(126,31,142,255))
    elif count == 120: temp_block = Image.new("RGBA",(64,64),(0,114,189,255))
    else :
        sims = [compare_images_pixel(f'testpic/{count}.png', f'testpic/{i}.png') for i in [0, 10, 110, 120]]
        weights = calculate_weights(sims)
        tempR = int(weights[0] * 217 + weights[1] * 237 + weights[2] * 126)
        tempG = int(weights[0] * 83 + weights[1] * 177 + weights[2] * 31 + weights[3] * 114)
        tempB = int(weights[0] * 25 + weights[1] * 32 + weights[2] * 142 + weights[3] * 189)
        temp_block = Image.new("RGBA",(64,64),(tempR , tempG , tempB , 255))

    color.append(temp_block)
    count += 1


def plot_arrow(A , B ):
    # 创建空白图像
    img = Image.new("RGBA", (64, 64), (255,255,255,0))
    draw = ImageDraw.Draw(img)

    # 箭头的起点坐标
    x0, y0 = 32, 32

    # 计算箭头的长度和方向
    length = np.sqrt(A ** 2 + B ** 2)
    direction = np.arctan2(B, A)

    # 箭头的终点坐标
    x1 = x0 + length * np.cos(direction)
    y1 = y0 + length * np.sin(direction)

    # 绘制箭头
    draw.line([(x0, y0), (x1, y1)], fill=(255,255,255,255), width=4)
    draw.polygon([(x1, y1), (x1 + 8 * np.cos(direction + np.pi * 3 / 4), y1 + 8 * np.sin(direction + np.pi * 3 / 4)),
                  (x1 + 8 * np.cos(direction - np.pi * 3 / 4), y1 + 8 * np.sin(direction - np.pi * 3 / 4))],
                 fill=(255,255,255,255))

    return img

arrow = []
count = 0
while count <= 120:
    if count == 0 : temp_block = plot_arrow( -30 , -30 )
    elif count == 10: temp_block = plot_arrow(30 , -30 )
    elif count == 110: temp_block = plot_arrow( -30 , 30 )
    elif count == 120: temp_block = plot_arrow( 30 , 30 )
    else :
        sims = [compare_images_pixel(f'testpic/{count}.png', f'testpic/{i}.png') for i in [0, 10, 110, 120]]
        weights = calculate_weights(sims)
        tempA = ( - weights[0] - weights[2] + weights[1] + weights[3] ) * 30
        tempB = ( - weights[0] + weights[2] - weights[1] + weights[3]) * 30
        temp_block = plot_arrow(tempA , tempB)
    arrow.append(temp_block)
    count += 1

big_image = Image.new("RGBA" , (11*64,11*64) , color = (255,255,255,0))


# for i in range(11):
#     for j in range(11):
#         # 获取当前位置的 RGB 图像
#         rgb_image = color[i * 11 + j]
#
#         # 计算当前图像在大图中的位置
#         position = (j * 64, i * 64)
#
#         # 将 RGB 图像粘贴到大图中
#         big_image.paste(rgb_image, position)
#
# for i in range(11):
#     for j in range(11):
#         # 获取当前位置的 RGB 图像
#         rgb_image = arrow[i * 11 + j]
#
#         # 计算当前图像在大图中的位置
#         position = (j * 64, i * 64)
#
#         # 将 RGB 图像粘贴到大图中
#         big_image.paste(rgb_image, position)

combined_images = []
for color_img, arrow_img in zip(color, arrow):
    # 创建一个空白图像，大小与color_img和arrow_img相同
    combined_img = Image.new("RGBA", color_img.size)

    # 将color_img和arrow_img进行合成
    combined_img = Image.alpha_composite(color_img, arrow_img)

    combined_images.append(combined_img)

for i in range(11):
    for j in range(11):
        # 获取当前位置的合成图像
        combined_img = combined_images[i * 11 + j]

        # 计算当前图像在大图中的位置
        position = (j * 64, i * 64)

        # 将合成图像粘贴到大图中
        big_image.paste(combined_img, position)


big_image.save("combinedtest_jian.png")