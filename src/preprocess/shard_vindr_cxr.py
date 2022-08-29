import os 

import pandas as pd
import numpy as np
import webdataset as wds
import random
from PIL import Image
import pydicom as dicom

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def readfile(file_path): #converting dicom to jpg
    img = dicom.dcmread(file_path)
    img = img.pixel_array
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img= Image.fromarray(np.uint8(img * 255), 'L')
    return img

def write_vindr_cxr_shards(out_pattern, df, set, maxcount=1000):
    with wds.ShardWriter(out_pattern, maxcount=maxcount, maxsize=10e9) as sink:
        for idx in range(df.shape[0]):
            img_name = set + df.loc[idx, "image_id"] + ".dicom"
            meta_cols= [x for x in df.columns if x not in ["image_id", "rad_id"]]
            image = readfile(os.path.join("/Volumes/TOSHIBA EXT/DATASETS/physionet.org/files/vindr-cxr/1.0.0/", img_name))
            meta = {col: str(df.loc[idx, col]) for col in meta_cols}

            sample = {
                "__key__": "%06d"%idx, 
                "jpg": image,
                "json": meta,
            }
            sink.write(sample)


if __name__ == "__main__":
    seed_everything(42)

    path = "/Volumes/TOSHIBA EXT/DATASETS/physionet.org/files/vindr-cxr/1.0.0/"
    train_df = pd.read_csv(os.path.join(path, 'annotations/image_labels_train.csv'))
    dev_df = pd.read_csv(os.path.join(path, 'annotations/image_labels_test.csv'))

    # shuffle train_df before sharding
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    #train and test df have different names for the "Other diseases" column
    dev_df = dev_df.rename(columns={'Other disease': 'Other diseases'})

    for df in [train_df, dev_df]:

        # handle Nans
        df["No finding"].fillna(0, inplace=True)
        df["Aortic enlargement"].fillna(0, inplace=True) 
        df["Atelectasis"].fillna(0, inplace=True) 
        df["Calcification"].fillna(0, inplace=True) 
        df["Cardiomegaly"].fillna(0, inplace=True)
        df["Clavicle fracture"].fillna(0, inplace=True)
        df["Consolidation"].fillna(0, inplace=True)
        df["Edema"].fillna(0, inplace=True)
        df["Emphysema"].fillna(0, inplace=True)
        df["Enlarged PA"].fillna(0, inplace=True)
        df["ILD"].fillna(0, inplace=True)
        df["Infiltration"].fillna(0, inplace=True)
        df["Lung Opacity"].fillna(0, inplace=True)
        df["Lung cavity"].fillna(0, inplace=True)
        df["Lung cyst"].fillna(0, inplace=True)
        df["Mediastinal shift"].fillna(0, inplace=True)
        df["Nodule/Mass"].fillna(0, inplace=True)
        df["Pleural effusion"].fillna(0, inplace=True)
        df["Pleural thickening"].fillna(0, inplace=True)
        df["Pneumothorax"].fillna(0, inplace=True)
        df["Pulmonary fibrosis"].fillna(0, inplace=True)
        df["Rib fracture"].fillna(0, inplace=True)
        df["Other lesion"].fillna(0, inplace=True)
        df["COPD"].fillna(0, inplace=True)
        df["Lung tumor"].fillna(0, inplace=True)
        df["Pneumonia"].fillna(0, inplace=True)
        df["Tuberculosis"].fillna(0, inplace=True)
        df["Other diseases"].fillna(0, inplace=True)

    write_vindr_cxr_shards(
        "/Volumes/Extreme SSD/sharded_datasets/vindr_cxr/train/vindr_cxr-train-%06d.tar", 
        train_df, "train/", maxcount=1000)
    
    # we don't really need shards for valid_df (N = 234)
    # but better to have this to maintain uniformity in 
    # training code
    write_vindr_cxr_shards(
        "/Volumes/Extreme SSD/sharded_datasets/vindr_cxr/val/vindr_cxr-dev-%06d.tar", 
        dev_df, "test/", maxcount=dev_df.shape[0])

