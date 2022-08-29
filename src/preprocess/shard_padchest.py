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
        
def write_padchest_shards(out_pattern, df, maxcount=1000, maxsize=(10e9)):
    with wds.ShardWriter(out_pattern, maxcount=maxcount, maxsize=maxsize) as sink:
        for idx in range(df.shape[0]):
            img_path = df.loc[idx, "path"]
            meta_cols = [x for x in df.columns if x != "path"]
            image = cv2.imread(os.path.join("/scratch/cs/chest-xray-ssl/BIMCV-PadChest-FULL/images", img_path))

            meta = {col: str(df.loc[idx, col]) for col in meta_cols}

            sample = {
                "__key__": "%06d"%idx, 
                "png": image,
                "json": meta,
            }
            try:
                sink.write(sample)
            except (SyntaxError, AssertionError) as e:
                print(os.path.join("/scratch/cs/chest-xray-ssl/BIMCV-PadChest-FULL/images", img_path))
                continue


if __name__ == "__main__":
    seed_everything(42)

    padchest_path = "/scratch/cs/chest-xray-ssl/BIMCV-PadChest-FULL/" 
    df = pd.read_csv(os.path.join(padchest_path, 'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv'), index_col=0)
    
    #dummy label for nans
    df["Labels"].fillna("['missing label']", inplace=True)
    
    #path creation
    df["path"] = df[["ImageDir", "ImageID"]].apply(lambda x: '/'.join(x.astype(str)), axis=1)
    

    # shuffle train_df before sharding
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    write_padchest_shards(
        "/scratch/cs/chest-xray-ssl/sharded_datasets/padchest/padchest-data-%06d.tar", 
        df, maxcount=1000, maxsize=10e9
    )
    
