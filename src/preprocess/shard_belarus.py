import os 
import glob
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

def write_belarus_shards(out_pattern, indexes, image_paths, classes, maxsize=(10e9)):
    with wds.ShardWriter(out_pattern, maxsize=maxsize) as sink:
        for idx in indexes:
            image_filepath = image_paths[idx]
            image = readfile(image_filepath)

            #basename = image_filepath.split('/')[-1]
            label = image_filepath.split('/')[-2] #find_label(basename, df_lab)
            meta = {col: 0.0 for col in classes}
            meta[label] = 1.0

            sample = {
                "__key__": "%06d"%idx, 
                "jpg": image,
                "json": meta,
            }

            sink.write(sample)

if __name__ == "__main__":
    seed_everything(42)
    
    image_paths = [] #to store image paths in list
    classes = [] #to store class values
    path = "/Volumes/TOSHIBA EXT/DATASETS/belarus"

    for path in glob.glob(path + '/*'):
        classes.append(path.split('/')[-1]) 
        image_paths.append(glob.glob(path + '/*'))

    image_paths = list(np.concatenate(image_paths))
    #Shuffling indexes
    indexes = list(range(len(image_paths)))
    random.shuffle(indexes)

    write_belarus_shards(
        "/Volumes/TOSHIBA EXT/DATASETS/sharded_datasets/belarus/belarus-data-%06d.tar", 
        indexes, image_paths, classes, maxsize=10e9)
    