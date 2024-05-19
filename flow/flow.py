import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.transforms import AffineTransform
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple
import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import random

class CustomDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 split: str,
                 transform: Callable,
                 target_size: Tuple[int, int],
                 **kwargs):
        self.data_dir = Path(data_path)  # / "OxfordPets"
        self.transforms = transform
        self.target_size = target_size
        # self.transforms = transforms.Grayscale(num_output_channels = 1)
        # imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.png'])
        random.shuffle(imgs)
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("L")  # 转换为单通道（灰度图）

        # 调整图像大小
        img = img.resize(self.target_size, Image.BILINEAR)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy data to prevent breaking



torch.manual_seed(0)
target_image_size = (28, 28)
train_data = CustomDataset('../ttf/simple',
                            split='train',
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5]),


                            ]),
                           target_size = target_image_size,)

test_data = CustomDataset('../ttf/simple',
                           split='test',
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.5], [0.5]),

                           ]),
                          target_size=target_image_size,)


class StandardLogisticDistribution:

    def __init__(self, data_dim=28 * 28, device='cuda'):
        self.m = TransformedDistribution(
            Uniform(torch.zeros(data_dim, device=device),
                    torch.ones(data_dim, device=device)),
            [SigmoidTransform().inv, AffineTransform(torch.zeros(data_dim, device=device),
                                                     torch.ones(data_dim, device=device))]
        )

    def log_pdf(self, z):
        return self.m.log_prob(z).sum(dim=1)

    def sample(self):
        return self.m.sample()


class NICE(nn.Module):

    def __init__(self, data_dim=28 * 28, hidden_dim=1000):
        super().__init__()

        self.m = torch.nn.ModuleList([nn.Sequential(
            nn.Linear(data_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, data_dim // 2), ) for i in range(4)])
        self.s = torch.nn.Parameter(torch.randn(data_dim))

    def forward(self, x):
        x = x.clone()
        for i in range(len(self.m)):
            x_i1 = x[:, ::2] if (i % 2) == 0 else x[:, 1::2]
            x_i2 = x[:, 1::2] if (i % 2) == 0 else x[:, ::2]
            h_i1 = x_i1
            h_i2 = x_i2 + self.m[i](x_i1)
            x = torch.empty(x.shape, device=x.device)
            x[:, ::2] = h_i1
            x[:, 1::2] = h_i2
        z = torch.exp(self.s) * x
        log_jacobian = torch.sum(self.s)
        return z, log_jacobian

    def invert(self, z):
        x = z.clone() / torch.exp(self.s)
        for i in range(len(self.m) - 1, -1, -1):
            h_i1 = x[:, ::2]
            h_i2 = x[:, 1::2]
            x_i1 = h_i1
            x_i2 = h_i2 - self.m[i](x_i1)
            x = torch.empty(x.shape, device=x.device)
            x[:, ::2] = x_i1 if (i % 2) == 0 else x_i2
            x[:, 1::2] = x_i2 if (i % 2) == 0 else x_i1
        return x


def training(normalizing_flow, optimizer, dataloader, distribution, nb_epochs=1500, device='cuda'):
    training_loss = []
    for _ in tqdm(range(nb_epochs)):

        for batch, _ in dataloader:
            batch = batch.view(-1, 28 * 28)
            z, log_jacobian = normalizing_flow(batch.to(device))
            log_likelihood = distribution.log_pdf(z) + log_jacobian
            loss = -log_likelihood.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

    return training_loss


if __name__ == '__main__':
    device = 'cuda'
    normalizing_flow = NICE().to(device)
    logistic_distribution = StandardLogisticDistribution(device=device)
    x = torch.randn(10, 28 * 28, device=device)
    assert torch.allclose(normalizing_flow.invert(normalizing_flow(x)[0]), x, rtol=1e-04, atol=1e-06)
    optimizer = torch.optim.Adam(normalizing_flow.parameters(), lr=0.0002, weight_decay=0.9)
    dataloader = DataLoader(train_data,
                            batch_size=32,
                            shuffle=True)
    training_loss = training(normalizing_flow,
                             optimizer,
                             dataloader,
                             logistic_distribution,
                             nb_epochs=500,
                             device=device)

    nb_data = 10
    fig, axs = plt.subplots(nb_data, nb_data, figsize=(10, 10))
    for i in range(nb_data):
        for j in range(nb_data):
            x = normalizing_flow.invert(logistic_distribution.sample().unsqueeze(0)).data.cpu().numpy()
            axs[i, j].imshow(x.reshape(28, 28).clip(0, 1), cmap='gray')
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    plt.savefig('simpleflow.png')
    plt.show()
