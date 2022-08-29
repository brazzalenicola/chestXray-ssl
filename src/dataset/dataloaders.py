import os

import dataset.chexpert_dataloaders as chexpert
import dataset.mimic_dataloaders as mimic
from torch.utils.data import ChainDataset

def get_mixed_dataloaders(
    shard_root_path,
    batch_size, 
    train_transofmer,
    val_transofrmer=None,
    img_size=512,
    num_workers=4
):

    chexpert_trainloader, chexpert_validloader, _ = chexpert.get_chexpert_dataloaders(shard_root_path, batch_size=batch_size,
        train_transofmer=train_transofmer,
        val_transofrmer=None,
        img_size=img_size,
        num_workers=num_workers)

    mimic_trainloader, mimic_validloader = mimic.get_mimic_dataloaders(shard_root_path, batch_size=batch_size,
        train_transofmer=train_transofmer,
        img_size=img_size,
        num_workers=num_workers)
    """
    padchest_trainloader, padchest_validloader = padchest.get_padchest_dataloaders(os.path.join(shard_root_path, "padchest"), batch_size=batch_size,
        train_transofmer=train_transofmer,
        val_transofrmer=val_transofrmer,
        img_size=img_size,
        num_workers=num_workers)

    nih_trainloader, nih_validloader = nih.get_nih_dataloaders(os.path.join(shard_root_path, "nih"), batch_size=batch_size,
        train_transofmer=train_transofmer,
        val_transofrmer=val_transofrmer,
        img_size=img_size,
        num_workers=num_workers)
    
    vindr_trainloader, vindr_validloader = vindr.get_vindr_dataloaders(os.path.join(shard_root_path, "vindr_cxr"), batch_size=batch_size,
        train_transofmer=train_transofmer,
        val_transofrmer=val_transofrmer,
        img_size=img_size,
        num_workers=num_workers)
    """
    train_loaders = ChainDataset([mimic_trainloader, chexpert_trainloader])
    #train_loaders = CombinedLoader([chexpert_trainloader, mimic_trainloader], "max_size_cycle")
    #train_loaders = [mimic_trainloader, chexpert_trainloader]
    val_loader = [chexpert_validloader, mimic_validloader]

    return train_loaders, val_loader
