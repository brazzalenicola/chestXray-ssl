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
    lbl =  np.array(df[df["img_name"] == name]["class"])
    if lbl.size == 0:
        return ""
    return lbl[0]

def write_extensive_shards(out_pattern, indexes, image_paths, df, classes, maxsize=(10e9)):
    with wds.ShardWriter(out_pattern, maxsize=maxsize) as sink:
        for idx in indexes:
            image_filepath = image_paths[idx]
            image = readfile(image_filepath)

            basename = image_filepath.split('/')[-1]
            label = find_label(basename, df) 

            if not label: #in case the loop put in the image_paths list files that are not images
                continue
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
    
    dataset_path = 'Extensive CXR/' 
    image_paths = [] #to store image paths in list
    classes = [] #to store class values

    for root, subdirectories, files in os.walk(dataset_path):
        if subdirectories != "meta":
            for file in files:
                if not file.startswith('.'):
                    image_paths.append(os.path.join(root, file))

    #Shuffling indexes
    indexes = list(range(len(image_paths)))
    random.shuffle(indexes)

    #dataframes with metadata, dont know why they are so many
    #put them all together so I dont have to look up the right df for every single img

    df_lab = pd.read_csv("/Volumes/TOSHIBA EXT/DATASETS/Extensive CXR/meta/additional_covid_metadata.csv")
    df_lab = df_lab.append(pd.read_csv("/Volumes/TOSHIBA EXT/DATASETS/Extensive CXR/meta/covid_metadata_ijcar.csv"))
    df_lab = df_lab.append(pd.read_csv("/Volumes/TOSHIBA EXT/DATASETS/Extensive CXR/meta/covid_metadata.csv"))
    df_lab = df_lab.append(pd.read_csv("/Volumes/TOSHIBA EXT/DATASETS/Extensive CXR/meta/normal_metadata_ijcar.csv"))
    df_lab = df_lab.append(pd.read_csv("/Volumes/TOSHIBA EXT/DATASETS/Extensive CXR/meta/normal_metadata.csv"))
    df_lab = df_lab.append(pd.read_csv("/Volumes/TOSHIBA EXT/DATASETS/Extensive CXR/meta/pneumonia_metadata_ijcar.csv"))
    df_lab = df_lab.append(pd.read_csv("/Volumes/TOSHIBA EXT/DATASETS/Extensive CXR/meta/pneumonia_metadata.csv"))

    classes = np.unique(df_lab["class"])

    write_extensive_shards(
        "/Volumes/TOSHIBA EXT/DATASETS/sharded_datasets/Extensive CXR/extensive-data-%06d.tar", 
        indexes, image_paths, df_lab, classes, maxsize=10e9)
    