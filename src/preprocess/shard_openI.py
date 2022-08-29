import os 
import glob
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

def write_openi_shards(out_pattern, indexes, image_paths, classes, maxsize=(10e9)):
    with wds.ShardWriter(out_pattern, maxsize=maxsize) as sink:
        for idx in indexes:
            image_filepath = image_paths[idx]
            image = readfile(image_filepath)

            meta = {"cxr": 1.0}

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
    path = "/Volumes/TOSHIBA EXT/DATASETS/OpenI/NLMCXR_png"

    dataset_path = 'OpenI/NLMCXR_png' 
    image_paths = [] #to store image paths in list
    classes = [] #to store class values

    for data_path in glob.glob(dataset_path + '/*'):
        classes.append(data_path.split('/')[-1]) 
        image_paths.append(glob.glob(data_path + '/*'))

    image_paths = list(np.concatenate(image_paths))
    #Shuffling indexes
    indexes = list(range(len(image_paths)))
    random.shuffle(indexes)

    write_openi_shards(
        "/Volumes/Extreme SSD/sharded_datasets/OpenI/openi-data-%06d.tar", 
        indexes, image_paths, classes, maxsize=10e9)
    