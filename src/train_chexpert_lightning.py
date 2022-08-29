from asyncio.log import logger
from calendar import firstweekday
import os

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import torch
import timm
import torchvision.transforms as transforms
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

import warnings
warnings.simplefilter("ignore")

import utils.common as common
import dataset.chexpert_dataloaders as loaders
from models.resnet50 import Resnet50
from models.resnet18 import Resnet18
from models.densenet121 import Densenet121
from models.googlenet import GoogleNet
from models.vgg16 import VGG16
from models.alexnet import AlexNet
from models.efficientnet import EfficientNet
from models.simCLR import SimCLR
from models.barlowTwins import BarlowTwins
from models.vit import VisionTransformer

from models.barlowTwins import GaussianBlur, Solarization, CLAHE

import wandb
wandb.init(entity="aalto-ml4h", project="chest-xray-ssl")

MODEL_TYPE = {
    "resnet50": Resnet50,
    "resnet18": Resnet18,
    "densenet121": Densenet121,
    "googlenet": GoogleNet,
    "vgg16" : VGG16,
    "alexnet" : AlexNet,
    "efficientnet" : EfficientNet,
    "vit" : VisionTransformer,
}

def load_barlowTwins_checkpoint(config, filepath=None):

    encoder = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
    ckpt_model = BarlowTwins(encoder=encoder, encoder_out_dim=1792, num_training_samples=580000, batch_size=config.TRAIN.BATCH_SIZE, z_dim=2048)
    # Load best checkpoint after training
    checkpoint = torch.load(filepath)
    ckpt_model.load_state_dict(checkpoint['state_dict'])

    return ckpt_model.encoder

def load_simclr_checkpoint(config, filepath=None):

    ckpt_model = SimCLR(hidden_dim=128, lr=config.TRAIN.LR, temperature=0.07, weight_decay=config.TRAIN.WEIGHT_DECAY)
    # Load best checkpoint after training
    checkpoint = torch.load(filepath)
    ckpt_model.load_state_dict(checkpoint['state_dict'])

    return ckpt_model.convnet

def get_train_transforms_clahe(img_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((img_size,img_size), scale=(0.85, 0.95)),
        CLAHE(p=0.8),
        GaussianBlur(p=1.0),
        Solarization(p=0.0),
        transforms.ToTensor(),
    ])
    return transform

def get_test_transforms_clahe(img_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((img_size,img_size), scale=(0.85, 0.95)),
        CLAHE(p=1.0),
        GaussianBlur(p=0.1),
        Solarization(p=0.4),
        transforms.ToTensor(),
    ])
    return transform

def train(config, model, train_loader, val_loader, max_epochs=20):

    wandb_logger = WandbLogger(project="chest-xray-ssl")

    trainer = pl.Trainer(
        default_root_dir= config.PATHS.MODEL_OUT_DIR,
        gpus = 1 if torch.cuda.is_available() else 0,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss", save_last=True)
        ],
        logger=wandb_logger,
        progress_bar_refresh_rate=1,
    )
    trainer.logger._default_hp_metric = None  

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load best checkpoint after training
    trained_model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    return trained_model, trainer


def main():
    args = common.parse_args()
    config = common.handle_config_and_log_paths(args)

    os.makedirs(os.path.join(config.PATHS.MODEL_OUT_DIR), exist_ok=True)
    common.seed_everything(seed=config.SEED)


    #train_transform = get_train_transforms_clahe(config.TRAIN.IMG_SIZE)
    #val_transform = get_test_transforms_clahe(config.TRAIN.IMG_SIZE)

    chexpert_trainloader, chexpert_validloader, chexpert_testloader = loaders.get_chexpert_dataloaders(
        shard_root_path=config.PATHS.SHARD_ROOT,
        batch_size=config.TRAIN.BATCH_SIZE,
        train_transofmer=None,
        val_transofrmer=None,
        img_size=config.TRAIN.IMG_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS
    )
    
    ckpt_model = load_barlowTwins_checkpoint(config, filepath="/scratch/work/brazzan1/chest-xray-ssl/chest-xray-ssl/saved_models/experiments/efficientnetb4_barlowTwin/at_2022_03_01_11_29_53/saved_models/lightning_logs/version_66958068/checkpoints/epoch=6-step=21482.ckpt")

    eff_model = MODEL_TYPE[config.TRAIN.MODEL_NAME](num_classes=5, lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY, pretrained_model = ckpt_model)
    
    trained_model, trainer = train(
        config,
        eff_model, 
        train_loader=chexpert_trainloader, 
        val_loader=chexpert_validloader, 
        max_epochs=config.TRAIN.N_EPOCHS, 
    )

    # test acc
    trainer.test(trained_model, dataloaders=chexpert_testloader)

if __name__ == "__main__":
    main()



