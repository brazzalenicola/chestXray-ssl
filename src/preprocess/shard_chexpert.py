import os 

import pandas as pd
import numpy as np
import webdataset as wds
import random

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def readfile(file_path):
    with open(file_path, "rb") as stream:
        return stream.read()

def write_chexpert_shards(out_pattern, df, maxcount=10000):
    with wds.ShardWriter(out_pattern, maxcount=maxcount) as sink:
        for idx in range(df.shape[0]):
            img_path = df.loc[idx, "Path"]
            meta_cols= [x for x in df.columns if x != "Path"]
            image = readfile(os.path.join("../../input", img_path))
            meta = {col: str(df.loc[idx, col]) for col in meta_cols}

            sample = {
                "__key__": "%06d"%idx, 
                "jpg": image,
                "json": meta,
            }
            sink.write(sample)


if __name__ == "__main__":
    seed_everything(42)

    chexpert_path = "../../input/CheXpert-v1.0-small/"
    train_df = pd.read_csv(os.path.join(chexpert_path, 'train.csv'))
    dev_df = pd.read_csv(os.path.join(chexpert_path, 'valid.csv'))

    # shuffle train_df before sharding
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    for df in [train_df, dev_df]:

        # store patient id in a separate column
        df["patient_id"] = df["Path"].apply(lambda row: "_".join(row.split("/")[2:4]))

        # handle Nans
        df["No Finding"].fillna(0, inplace=True)
        df["Enlarged Cardiomediastinum"].fillna(0, inplace=True)
        df["Cardiomegaly"].fillna(0, inplace=True)
        df["Lung Opacity"].fillna(0, inplace=True)
        df["Lung Lesion"].fillna(0, inplace=True)
        df["Edema"].fillna(0, inplace=True)
        df["Consolidation"].fillna(0, inplace=True)
        df["Pneumonia"].fillna(0, inplace=True)
        df["Atelectasis"].fillna(0, inplace=True)
        df["Pneumothorax"].fillna(0, inplace=True)
        df["Pleural Effusion"].fillna(0, inplace=True)
        df["Pleural Other"].fillna(0, inplace=True)
        df["Fracture"].fillna(0, inplace=True)
        df["Support Devices"].fillna(0, inplace=True)

    write_chexpert_shards(
        "../../input/chexpert-shards/chexpert-train-%06d.tar", 
        train_df, maxcount=10000
    )
    
    # we don't really need shards for valid_df (N = 234)
    # but better to have this to maintain uniformity in 
    # training code
    write_chexpert_shards(
        "../../input/chexpert-shards/chexpert-dev-%06d.tar", 
        dev_df, maxcount=dev_df.shape[0]
    )

