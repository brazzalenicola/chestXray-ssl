import os 
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

def write_siim_acr_shards(out_pattern, indexes, image_paths, maxsize=(10e9)):
    with wds.ShardWriter(out_pattern, maxsize=maxsize) as sink:
        for idx in indexes:
            image_filepath = image_paths[idx]
            image = readfile(image_filepath)

            label = 'pneumothorax'
            meta = {label: 1.0}

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
    path = "/Volumes/TOSHIBA EXT/DATASETS/siim-acr/"

    for root, subdirectories, files in os.walk(path):
        for file in files:
            if not file.startswith('.'): #to exclude .DS_Store
                image_paths.append(os.path.join(root, file))

    #Shuffling indexes
    indexes = list(range(len(image_paths)))
    random.shuffle(indexes)

    write_siim_acr_shards(
        "/Volumes/Extreme SSD/sharded_datasets/siim-acr/siim-acr-data-%06d.tar", 
        indexes, image_paths, maxsize=10e9)
    