import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import webdataset as wds
import numpy as np

def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_train_transforms(img_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def get_test_transforms(img_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def get_padchest_dataloaders(
    shard_root_path,
    batch_size, 
    train_transofmer,
    val_transofmer,
    img_size=64,
    num_workers=4
):

    transform = get_train_transforms(img_size)
    ts_transform = get_test_transforms(img_size)

    n_shards = len([p for p in os.listdir(shard_root_path) if 'padchest' in p])

    #currently using 10% as validation, can be changed
    train_start = str(0).zfill(6)
    train_end = str(int(np.floor(n_shards*0.9))-1).zfill(6)
    # train_end = str(n_shards-2).zfill(6)
    val_start = str(int(np.floor(n_shards*0.9))).zfill(6)
    val_end = str(n_shards-1).zfill(6)


    # ref: https://stackoverflow.com/a/42521252
    train_shards = os.path.join(
        shard_root_path, f"padchest-data-{{{train_start}..{train_end}}}.tar")
    val_shard = os.path.join(
        shard_root_path, f"padchest-data-{{{val_start}..{val_end}}}.tar")      

    train_dataset = wds.WebDataset(train_shards)\
        .shuffle(1000)\
        .decode("pil")\
        .rename(image="jpg;png", meta="json")\
        .map_dict(image=transform)\
        .to_tuple("image", "meta")

    val_dataset = wds.WebDataset(val_shard)\
        .shuffle(1000)\
        .decode("pil")\
        .rename(image="jpg;png", meta="json")\
        .map_dict(image=ts_transform)\
        .to_tuple("image", "meta")            

    padchest_trainloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    padchest_validloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers
    )

    return padchest_trainloader, padchest_validloader

