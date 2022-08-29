import os
from PIL import Image
import torch
import braceexpand
import torchvision.transforms as transforms
import webdataset as wds

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

def get_mimic_dataloaders(
    shard_root_path,
    batch_size, 
    train_transofmer,
    img_size=64,
    num_workers=4
):
    shard_root_path = os.path.join(shard_root_path, "mimic")
    # we use one shard for validation and the rest for testing
    # TODO: maybe need to rethink this, currently 10K valid examples 
    n_shards_train = len([p for p in os.listdir(os.path.join(shard_root_path, "train")) if 'train' in p])
    n_shards_val = len([p for p in os.listdir(os.path.join(shard_root_path, "val")) if 'val' in p])
    n_shards_test = len([p for p in os.listdir(os.path.join(shard_root_path, "test")) if 'test' in p])

    train_start = str(0).zfill(6)
    train_end = str(n_shards_train-1).zfill(6)
    # train_end = str(n_shards-2).zfill(6)
    val_start = str(0).zfill(6)
    val_end = str(n_shards_val-1).zfill(6)
    test_start = str(0).zfill(6)
    test_end = str(n_shards_test-1).zfill(6)

    # ref: https://stackoverflow.com/a/42521252
    train_shards = os.path.join(
        shard_root_path, "train" ,f"mimic-train-{{{train_start}..{train_end}}}.tar")

    #for ssl we dont need a test set, so I put val and test together
    val_shard = (list(braceexpand.braceexpand(os.path.join( shard_root_path, "val" ,f"mimic-val-{{{val_start}..{val_end}}}.tar"))) + 
    list(braceexpand.braceexpand(os.path.join( shard_root_path, "test" , f"mimic-test-{{{test_start}..{test_end}}}.tar"))))
        

    train_dataset = wds.WebDataset(train_shards)\
        .shuffle(1000)\
        .decode("pil")\
        .rename(image="jpg;png", meta="json")\
        .map_dict(image=train_transofmer)\
        .to_tuple("image", "meta")

    val_dataset = wds.WebDataset(val_shard)\
        .shuffle(1000)\
        .decode("pil")\
        .rename(image="jpg;png", meta="json")\
        .map_dict(image=train_transofmer)\
        .to_tuple("image", "meta")             

    mimic_trainloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    mimic_validloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_dataset, mimic_validloader

