from asyncio.log import logger
import os

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


import utils.common as common
import dataset.chexpert_dataloaders as loaders
from models.simCLR import SimCLR

import warnings
warnings.simplefilter("ignore")


def train_simclr(config, train_loader, val_loader, batch_size, max_epochs, model_suffix="", **kwargs):
    wandb_logger = WandbLogger(project="chest-xray-ssl")

    trainer = pl.Trainer(
        default_root_dir= config.PATHS.MODEL_OUT_DIR,
        gpus = 1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ],
        logger=wandb_logger,
        progress_bar_refresh_rate=1,
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    model = SimCLR(max_epochs=max_epochs, **kwargs)
    trainer.fit(model, train_loader, val_loader)
    # Load best checkpoint after training
    trained_model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return trained_model

def main():
    args = common.parse_args()
    config = common.handle_config_and_log_paths(args)

    os.makedirs(os.path.join(config.PATHS.MODEL_OUT_DIR), exist_ok=True)
    common.seed_everything(seed=config.SEED)

    mixed_trainloader, mixed_validloader = loaders.get_mixed_dataloaders(
    batch_size=config.TRAIN.BATCH_SIZE,
    img_size=config.TRAIN.IMG_SIZE,
    num_workers=config.TRAIN.NUM_WORKERS
    )

    simclr_model = train_simclr(
    config, mixed_trainloader, mixed_validloader, batch_size=config.TRAIN.BATCH_SIZE, hidden_dim=128, lr=config.TRAIN.LR, temperature=0.07, weight_decay=config.TRAIN.WEIGHT_DECAY, max_epochs=config.TRAIN.N_EPOCHS)

if __name__ == "__main__":
    main()
