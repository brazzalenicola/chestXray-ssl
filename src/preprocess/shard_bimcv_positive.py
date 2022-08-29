import os 

import pandas as pd
import numpy as np
import webdataset as wds
import random
import ast
import cv2
from PIL import Image

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def readfile(file_path):
    with open(file_path, "rb") as stream:
        return stream.read()
        
def write_bimcv_pos_shards(out_pattern, df, maxcount=1000, maxsize=(10e9)):
    with wds.ShardWriter(out_pattern, maxcount=maxcount, maxsize=maxsize) as sink:
        for idx in range(df.shape[0]):
            img_path = df.loc[idx, "path"]
            meta_cols = [x for x in df.columns if x != "path"]
            image = readfile(os.path.join("/scratch/cs/chest-xray-ssl/BIMCV-COVID19-cIter_1_2/covid19_posi", img_path))

            meta = {col: str(df.loc[idx, col]) for col in meta_cols}

            sample = {
                "__key__": "%06d"%idx, 
                "png": image,
                "json": meta,
            }

            sink.write(sample)


if __name__ == "__main__":
    seed_everything(42)

    bimcv_path = "/scratch/cs/chest-xray-ssl/BIMCV-COVID19-cIter_1_2" 

    df = pd.read_csv(os.path.join(bimcv_path, 'partitions_pos.csv'), sep=";", usecols=["subject", "session", "filepath"])
    df["path"] = df["filepath"].apply(lambda row: str(row) if "_ct" not in str(row) else None)
    df = df.mask(df.eq('None')).dropna()
    df["path"] = df["path"].apply(lambda row: str(row)[2:])
    
    df_labels = pd.read_csv(os.path.join(bimcv_path, 'labels_covid_posi.csv'), sep=";")
    df_unique = df.merge(df_labels, left_on="session", right_on="ReportID")
    df_unique.drop(columns=["Unnamed: 0", "filepath"], inplace=True)

    df_unique["Report"].fillna(0, inplace=True)
    df_unique["Labels"].fillna("['missing label']", inplace=True)

    df_unique = df_unique.mask(df_unique.eq('None')).dropna()

    # shuffle train_df before sharding
    df_unique = df_unique.sample(frac=1.0, random_state=42).reset_index(drop=True)

    write_bimcv_pos_shards(
        "/scratch/cs/chest-xray-ssl/sharded_datasets/bimcv_pos/bimcv_pos-data-%06d.tar", 
        df=df_unique, maxcount=1000, maxsize=10e9
    )
    
