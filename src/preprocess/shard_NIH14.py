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

def find_label(name, df):
    lbl =  np.array(df[df["Image Index"] == name]["Finding Labels"])
    label = lbl[0].split('|')
    return label

def write_NIH14_shards(out_pattern, indexes, image_paths, classes, df, maxcount=1000):
    with wds.ShardWriter(out_pattern, maxcount=maxcount) as sink:
        for idx in indexes:
            image_filepath = image_paths[idx]
            image = readfile(image_filepath)

            basename = image_filepath.split('/')[-1]
            label = find_label(basename, df) 
            
            meta = {col: 0.0 for col in classes}
            for l in label:
                meta[l] = 1.0

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
    path = "/Volumes/TOSHIBA EXT/DATASETS/NIH_CXR14/NIH_CXR14/"

    for path in glob.glob(path + '/*'):
        #classes.append(path.split('/')[-1]) 
        image_paths.append(glob.glob(path + '/*'))

    df_lab = pd.read_csv("/Volumes/TOSHIBA EXT/DATASETS/NIH_CXR14/Data_Entry_2017_v2020.csv")
    df_lab = df_lab.iloc[:,:2]
    labels = df_lab["Finding Labels"]

    #multi-label handling
    for l in labels:
        l = l.split('|')
        for t in l:
            if t not in classes:
                classes.append(t)
                
    image_paths = list(np.concatenate(image_paths))
    indexes = list(range(len(image_paths))) #shuffling indexes
    random.shuffle(indexes)

    write_NIH14_shards(
        "/Volumes/TOSHIBA EXT/DATASETS/sharded_datasets/NIH_CXR14/nih_cxr14-data-%06d.tar", 
        indexes, image_paths, classes, df_lab, maxcount=1000)


    