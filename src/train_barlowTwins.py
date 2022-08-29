from asyncio.log import logger
import os

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import utils.common as common
import dataset.dataloaders as loaders
from models.barlowTwins import BarlowTwins
from models.barlowTwins import BarlowTwinsTransform

import warnings
warnings.simplefilter("ignore")

import wandb
wandb.init(entity="aalto-ml4h", project="chest-xray-ssl")

def train_barlowTwins(config, encoder, train_loader, val_loader, batch_size, max_epochs=40, model_suffix="", **kwargs):

    model = BarlowTwins(
        encoder=encoder,
        encoder_out_dim=1792,
        num_training_samples=580000,
        batch_size=batch_size,
        z_dim=1500,
    )

    checkpoint_callback = ModelCheckpoint(every_n_val_epochs=1, save_top_k=1, save_last=True)
    wandb_logger = WandbLogger(project="chest-xray-ssl")

    trainer = pl.Trainer(
        default_root_dir= config.PATHS.MODEL_OUT_DIR,
        max_time="05:00:00:00",
        max_epochs=max_epochs,
        gpus= 4,
        strategy="dp",
        precision=16 if torch.cuda.device_count() > 0 else 32,
        callbacks=checkpoint_callback,
        logger=wandb_logger,
        progress_bar_refresh_rate=1,
    )

    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint after training
    trined_model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return trined_model

def main():
    args = common.parse_args()
    config = common.handle_config_and_log_paths(args)

    os.makedirs(os.path.join(config.PATHS.MODEL_OUT_DIR), exist_ok=True)
    common.seed_everything(seed=config.SEED)
    
    transform_t = BarlowTwinsTransform(input_height=config.TRAIN.IMG_SIZE, gaussian_blur=False, normalize=None)

    train, validloader = loaders.get_mixed_dataloaders(
    shard_root_path=config.PATHS.SHARD_ROOT,
    batch_size=config.TRAIN.BATCH_SIZE,
    train_transofmer=transform_t,
    img_size=config.TRAIN.IMG_SIZE,
    num_workers=config.TRAIN.NUM_WORKERS
    )

    trainloader = torch.utils.data.DataLoader(
        train, 
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS
    )

    encoder = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)

    barlowTwins_model = train_barlowTwins(
    config, encoder, trainloader, validloader, batch_size=config.TRAIN.BATCH_SIZE, hidden_dim=128, lr=config.TRAIN.LR, temperature=0.07, weight_decay=config.TRAIN.WEIGHT_DECAY, max_epochs=config.TRAIN.N_EPOCHS)

if __name__ == "__main__":
    main()
