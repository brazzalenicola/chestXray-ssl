from functools import partial

import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1792, hidden_dim=1729, output_dim=1024):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)


def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            gauss = transforms.GaussianBlur(kernel_size=3) 
            return gauss(img)
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return transforms.functional.solarize(img, threshold=128)
        else:
            return img

class CLAHE(object):
    def __init__(self, p):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize=(16,16))
            final_img = clahe.apply(img) #+ img.min()
            final_img = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
            return Image.fromarray(final_img)
        else: 
            return img


class CoarseDropout(object):
    def __init__(self,p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
            aug = iaa.CoarseDropout(0.05, size_percent=0.25)
            img = aug(images=img)
            img = (img - img.min())/(img.max()-img.min())
            img = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2BGR)
            return img
        else:
            return img

class BarlowTwinsTransform:
    def __init__(self, input_height=224, gaussian_blur=True, normalize=None):

        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop((self.input_height,self.input_height), scale=(0.85, 0.95)),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomGrayscale(p=0.2),
            CLAHE(p=0.9),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop((self.input_height,self.input_height), scale=(0.85, 0.95)),
            CLAHE(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, sample):
        y1 = self.transform(sample)
        y2 = self.transform_prime(sample)
        return y1, y2


class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=1024):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag

class BarlowTwins(pl.LightningModule):
    def __init__(
        self,
        encoder,
        encoder_out_dim,
        num_training_samples,
        batch_size,
        lambda_coeff=5e-3,
        z_dim=2048,
        learning_rate=1e-4,
        warmup_epochs=2,
        max_epochs=200,
    ):
        super().__init__()

        self.encoder = encoder
        self.projection_head = ProjectionHead(input_dim=encoder_out_dim, hidden_dim=encoder_out_dim, output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        self.train_iters_per_epoch = num_training_samples // batch_size

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        (x1, x2), _ = batch

        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))

        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
    