from ntpath import join
import os
from PIL import Image
import torch
import braceexpand
import torchvision.transforms as transforms
import webdataset as wds
from models.simCLR import ContrastiveTransformations, GaussianBlur
from models.barlowTwins import BarlowTwinsTransform
from models.barlowTwins import GaussianBlur, Solarization, CLAHE

def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_train_transforms(img_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((img_size,img_size), scale=(0.85, 0.95)),
        #CLAHE(p=1.0),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform

def get_test_transforms(img_size):
    transform = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.85, 0.95)),
        #CLAHE(p=1.0),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform


def get_simCLR_transform(img_size, s=1):

    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    contrast_transforms = transforms.Compose([transforms.RandomResizedCrop((img_size,img_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * img_size)),
                                            transforms.ToTensor()])

    return contrast_transforms

def collate_train(list_of_samples):
    #list_of_samples -> tuple of this kind (images, metas)
    
    chexpert_targets = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    u_one_features = ['Atelectasis', 'Edema']
    u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']
    
    imgs = [t[0] for t in list_of_samples]
    imgs = torch.stack(imgs, dim=0)

    metas = [t[1] for t in list_of_samples]
    labels = []

    for meta in metas:
        #replacing -1.0 to 1.0 for the u_one features lists
        for feat in u_one_features: 
            if float(meta[feat])==-1.0:
                meta[feat] = 1.0
                
        #replacing -1.0 to 0.0 for the u_zero features lists
        for feat in u_zero_features: 
            if float(meta[feat])==-1.0:
                meta[feat] = 0.0
                
        labels.append(torch.tensor([float(meta[t]) for t in chexpert_targets]))

    labels = torch.stack(labels, dim=0) 

    return imgs, labels

#test set do not contain uncertainty labels u, 1s and 0s only
def collate_test(list_of_samples):
    #list_of_samples -> tuple of this kind (images, metas)
    
    chexpert_targets = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    
    imgs = [t[0] for t in list_of_samples]
    imgs = torch.stack(imgs, dim=0)

    metas = [t[1] for t in list_of_samples]
    
    labels = []
    for meta in metas:
        labels.append(torch.tensor([float(meta[t]) for t in chexpert_targets]))

    labels = torch.stack(labels, dim=0) 

    return imgs, labels

def get_chexpert_dataloaders(
    shard_root_path,
    batch_size, 
    train_transofmer=None,
    val_transofrmer=None,
    img_size=64,
    num_workers=4
):
    shard_root_path = os.path.join(shard_root_path, "chexpert")
    train_transofmer = get_train_transforms(img_size)
    val_transofrmer = get_test_transforms(img_size)

    # we use one shard for validation and the rest for testing
    # TODO: maybe need to rethink this, currently 10K valid examples 
    n_shards = len([p for p in os.listdir(shard_root_path) if 'train' in p])

    train_start = str(0).zfill(6)
    train_end = str(n_shards-2).zfill(6)
    # train_end = str(n_shards-2).zfill(6)
    val_start = str(n_shards-1).zfill(6)

    # ref: https://stackoverflow.com/a/42521252
    train_shards = os.path.join(
        shard_root_path, f"chexpert-train-{{{train_start}..{train_end}}}.tar")
    val_shard = os.path.join(
        shard_root_path, f"chexpert-train-{val_start}.tar")
    test_shard = os.path.join(
        shard_root_path, f"chexpert-dev-{str(0).zfill(6)}.tar"
    )        

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

    test_dataset = wds.WebDataset(test_shard)\
        .shuffle(1000)\
        .decode("pil")\
        .rename(image="jpg;png", meta="json")\
        .map_dict(image=val_transofrmer)\
        .to_tuple("image", "meta")        

    chexpert_trainloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_train
    )
    chexpert_validloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_train
    )
    chexpert_testloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_test 
    )

    return chexpert_trainloader, chexpert_validloader, chexpert_testloader

