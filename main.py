from time import time

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from load_celebA import get_dataloader
from model import VAE

# Hyperparameters
n_epochs = 200
kl_weight = 0.00025
lr = 0.005


def loss_fn(y, y_hat, mean, logvar):
    recons_loss = F.mse_loss(y_hat, y)
    kl_loss = torch.mean(
        -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)
    loss = recons_loss + kl_loss * kl_weight
    return loss


def train(device, dataloader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset_len = len(dataloader.dataset)

    begin_time = time()
    # train
    for i in range(n_epochs):
        loss_sum = 0
        for x in dataloader:
            x = x.to(device)
            y_hat, mean, logvar = model(x)
            loss = loss_fn(x, y_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        loss_sum /= dataset_len
        training_time = time() - begin_time
        minute = int(training_time // 60)
        second = int(training_time % 60)
        print(f'epoch {i}: loss {loss_sum} {minute}:{second}')
        torch.save(model.state_dict(), 'model.pth')


def reconstruct(device, dataloader, model):
    model.eval()
    batch = next(iter(dataloader))
    x = batch[0:1, ...].to(device)
    output = model(x)[0]
    output = output[0].detach().cpu()
    input = batch[0].detach().cpu()
    combined = torch.cat((output, input), 1)
    img = ToPILImage()(combined)
    img.save('work_dirs/tmp.jpg')

def reconstruct_multiple(device, dataloader, model, num_samples_per_side=4):
    model.eval()
    samples = next(iter(dataloader))
    samples = samples[:num_samples_per_side ** 2].to(device)  # 将所有样本移动到GPU

    reconstructed_outputs = model(samples)
    # 如果模型输出是一个元组，选择合适的元素作为张量
    if isinstance(reconstructed_outputs, tuple):
        reconstructed_outputs = reconstructed_outputs[0]

    originals_grid = make_grid(samples.cpu(), nrow=num_samples_per_side)
    reconstructions_grid = make_grid(reconstructed_outputs.cpu(), nrow=num_samples_per_side)

    combined_grid = torch.cat((originals_grid, reconstructions_grid), dim=1)
    img = ToPILImage()(combined_grid)
    img.save('work_dirs/multiple_samples_reconstruction.jpg')

def generate(device, model):
    model.eval()
    output = model.sample(device)
    output = output[0].detach().cpu()
    img = ToPILImage()(output)
    img.save('work_dirs/tmp.jpg')


def main():
    device = 'cuda:0'
    dataloader = get_dataloader()

    model = VAE().to(device)

    # If you obtain the ckpt, load it
    model.load_state_dict(torch.load('model.pth', 'cuda:0'))

    # Choose the function
    # train(device, dataloader, model)
    reconstruct(device, dataloader, model)
    # reconstruct_multiple('cuda:0', dataloader, model, num_samples_per_side=4)
    # generate(device, model)


if __name__ == '__main__':
    main()