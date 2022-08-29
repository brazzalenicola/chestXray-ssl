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

def write_mimic_shards(out_pattern, df, shard_path, maxcount=1000):
    with wds.ShardWriter(out_pattern, maxcount=maxcount) as sink:
        for idx in range(df.shape[0]):
            img_path = df.loc[idx, "path"]
            meta_cols= [x for x in df.columns if x not in ['split', 'folder', 'path']]
            image = readfile(os.path.join(shard_path, img_path))
            meta = {col: str(df.loc[idx, col]) for col in meta_cols}

            sample = {
                "__key__": "%06d"%idx, 
                "jpg": image,
                "json": meta,
            }
            sink.write(sample)


if __name__ == "__main__":
    seed_everything(42)

    path = "/scratch/cs/chest-xray-ssl/MIMIC/files/mimic-cxr-jpg/2.0.0/"

    chexeprt_df = pd.read_csv(os.path.join(path, 'mimic-cxr-2.0.0-chexpert.csv'))
    split_df = pd.read_csv(os.path.join(path, 'mimic-cxr-2.0.0-split.csv'))

    df_main = chexeprt_df.merge(split_df, on="study_id")

    #path generation
    df_main["folder"] = "p"+ df_main["subject_id_x"].apply(lambda row: str(row)[:2])
    df_main["subject_id_x"] = "p"+ df_main["subject_id_x"].apply(lambda row: str(row))
    df_main["study_id"] = "s"+ df_main["study_id"].apply(lambda row: str(row))
    df_main["dicom_id"] = df_main["dicom_id"].apply(lambda row: str(row))+".jpg"

    df_main["path"] = df_main[["folder", "subject_id_x", "study_id", "dicom_id"]].apply(lambda x: '/'.join(x.astype(str)), axis=1)

    #handle nan
    df_main["No Finding"].fillna(0, inplace=True)
    df_main["Enlarged Cardiomediastinum"].fillna(0, inplace=True)
    df_main["Cardiomegaly"].fillna(0, inplace=True)
    df_main["Lung Opacity"].fillna(0, inplace=True)
    df_main["Lung Lesion"].fillna(0, inplace=True)
    df_main["Edema"].fillna(0, inplace=True)
    df_main["Consolidation"].fillna(0, inplace=True)
    df_main["Pneumonia"].fillna(0, inplace=True)
    df_main["Atelectasis"].fillna(0, inplace=True)
    df_main["Pneumothorax"].fillna(0, inplace=True)
    df_main["Pleural Effusion"].fillna(0, inplace=True)
    df_main["Pleural Other"].fillna(0, inplace=True)
    df_main["Fracture"].fillna(0, inplace=True)
    df_main["Support Devices"].fillna(0, inplace=True)

    train_df = df_main[df_main["split"]=="train"]
    val_df = df_main[df_main["split"]=="validate"].reset_index(drop=True)
    test_df = df_main[df_main["split"]=="test"].reset_index(drop=True)

    # shuffle train_df before sharding
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    shard_path = "/scratch/cs/chest-xray-ssl/MIMIC/files/mimic-cxr-jpg/2.0.0/files/"

    write_mimic_shards(
        "/scratch/cs/chest-xray-ssl/sharded_datasets/mimic/train/mimic-train-%06d.tar", 
        train_df, shard_path, maxcount=1000
    )

    write_mimic_shards(
        "/scratch/cs/chest-xray-ssl/sharded_datasets/mimic/val/mimic-val-%06d.tar", 
        val_df, shard_path, maxcount=1000
    )

    write_mimic_shards(
        "/scratch/cs/chest-xray-ssl/sharded_datasets/mimic/test/mimic-test-%06d.tar", 
        test_df, shard_path, maxcount=1000
    )
