import time
import os
import logging
import sys
import random
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics

import torch
from torch.cuda.amp import GradScaler, autocast

import warnings
warnings.simplefilter("ignore")

import utils.common as common
import dataset.mimic_dataloaders as loaders
from models.resnet50 import Resnet50
from models.resnet18 import Resnet18
from models.densenet121 import Densenet121
from models.alexnet import AlexNet
from models.googlenet import GoogleNet
from models.vgg16 import VGG16
# import wandb

MODEL_TYPE = {
    "resnet50": Resnet50,
    "resnet18": Resnet18,
    "densenet121": Densenet121,
    "alexnet" : AlexNet,
    "googlenet" : GoogleNet,
    "vgg16" : VGG16
}


def save_best_model(config, model, best_val_loss, valid_loss, model_suffix=""):
    if valid_loss <= best_val_loss:
        print(
            f"Validation loss Decreased from {best_val_loss} to {valid_loss}"
        )

        best_val_loss = valid_loss
        torch.save(
            model.state_dict(),
            os.path.join(config.PATHS.MODEL_OUT_DIR, f"{config.TRAIN.MODEL_NAME}-{model_suffix}.pth"),
        )
    return best_val_loss

def load_best_model(config, model, model_suffix=""):
    model_path = os.path.join(config.PATHS.MODEL_OUT_DIR, f"{config.TRAIN.MODEL_NAME}-{model_suffix}.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(config.DEVICE)
    model.eval()

    return model


def train(
    config,
    model, 
    trainloader, 
    validloader, 
    optimizer, 
    lr_scheduler,
    n_epochs, 
    print_freq=500,  
    verbose=False
):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    global_iter = 0
    best_loss = np.float("inf")
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        print(f"Epoch #: {epoch}")
        running_loss = 0.0
        for i, (imgs, labs) in enumerate(trainloader, 0):
            images = imgs.to(config.DEVICE)
            #meta = batch[1]
            #labels = torch.tensor([float(x) for x in meta["Pneumonia"]]).to(config.DEVICE)
            labels = labs.to(config.DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, _ = model(images)
            loss = criterion(outputs, labels) # account for -1 in label
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # print statistics
            running_loss += loss.item()
            if i % print_freq == print_freq-1:    # print every 500 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / print_freq))
                running_loss = 0.0
        val_loss, val_acc = eval_loop(config, validloader, model, verbose=verbose)
        if val_loss < best_loss:
            best_loss = save_best_model(config, model, best_loss, val_loss)

    print('Finished Training')
    return model, val_loss, val_acc   


def eval_loop(config, testloader, model, verbose=False):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss()
    y_trues, y_preds = [], []
    running_loss = 0.0
    ctr = 0
    total = 0.0
    correct = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for (imgs, labs) in testloader:
            ctr += 1
            images = imgs.to(config.DEVICE)
            #meta = batch[1]
            labels = labs.to(config.DEVICE)
            # calculate outputs by running images through the network 
            outputs, _ = model(images)
            loss = criterion(outputs, labels) # account for -1 in label
            running_loss += loss.item()
            # the class with the highest energy is what we choose as prediction
            #_, predicted = torch.max(outputs.data, 1)
            predicted = torch.sigmoid(outputs).cpu().detach().numpy()
            #round up and down to either 1 or 0
            predicted = np.round(outputs)
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()
            y_trues.extend(labels.cpu().detach().numpy().tolist())
            y_preds.extend(predicted.cpu().detach().numpy().tolist())
            #calculate how many images were correctly classified
        val_loss = running_loss / ctr
        val_acc =  (y_preds == y_trues).sum()/len(y_trues) #metrics.accuracy_score(y_trues, y_preds)

    if verbose:
        print(f"Accuracy: {val_acc:.3f}")
        print(f"F1 Score: {metrics.f1_score(y_trues, y_preds, average='macro'):.3f}")
        print()
    return val_loss, val_acc


def main():
    args = common.parse_args()
    config = common.handle_config_and_log_paths(args)

    os.makedirs(os.path.join(config.PATHS.MODEL_OUT_DIR), exist_ok=True)
    common.seed_everything(seed=config.SEED)

    chexpert_trainloader, chexpert_validloader, chexpert_testloader = loaders.get_chexpert_dataloaders(
        shard_root_path=config.PATHS.SHARD_ROOT,
        batch_size=config.TRAIN.BATCH_SIZE,
        img_size=config.TRAIN.IMG_SIZE,
        num_workers=config.TRAIN.NUM_WORKERS
    )

    model = MODEL_TYPE[config.TRAIN.MODEL_NAME](num_classes=5)
    model = model.to(config.DEVICE)
    optimizer, lr_scheduler = model.configure_optimizer(
        lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    
    trained_model = train(
        config,
        model, 
        trainloader=chexpert_trainloader, 
        validloader=chexpert_validloader, 
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        n_epochs=config.TRAIN.N_EPOCHS, 
        print_freq=500,
        verbose=config.VERBOSE
    )

    # test acc
    trained_model = MODEL_TYPE[config.TRAIN.MODEL_NAME](num_classes=3)
    trained_model = load_best_model(config, trained_model, model_suffix="")
    test_acc = eval_loop(config, chexpert_testloader, trained_model, verbose=True)


if __name__ == "__main__":
    main()



