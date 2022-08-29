import os 
import numpy as np
import webdataset as wds
import random
from PIL import Image
import torchxrayvision as xrv

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def write_xrv_covid_shards(out_pattern, dataset, classes, maxsize=(10e9)):
    with wds.ShardWriter(out_pattern, maxsize=maxsize) as sink:
        for i, data in enumerate(dataset):
            imag = data["img"][0]
            imag = (imag - np.min(imag)) / (np.max(imag) - np.min(imag))
            imag= Image.fromarray(np.uint8(imag * 255), 'L')

            lab = np.array(data["lab"], dtype=np.float)
            meta = dict(zip(classes, lab))

            sample = {
                "__key__": "%06d"%i, 
                "jpg": imag,
                "json": meta,
            }

            sink.write(sample)

if __name__ == "__main__":
    seed_everything(42)

    #git clone https://github.com/ieee8023/covid-chestxray-dataset
    d = xrv.datasets.COVID19_Dataset(imgpath="covid-chestxray-dataset/images/", csvpath="covid-chestxray-dataset/metadata.csv")
    
    # class values
    classes = d.pathologies

    write_xrv_covid_shards(
        "/Volumes/TOSHIBA EXT/DATASETS/sharded_datasets/xrv_covid/xrv_covid-%01d.tar", 
        d , classes, maxsize=10e9)
    