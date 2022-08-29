import torch, os
from datetime import datetime
from yacs.config import CfgNode as CN

# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

_C.SEED = 100
_C.USE_AMP = False
_C.VERBOSE = True

# Choose device type
_C.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config destination (in OUT_DIR)
_C.CFG_DEST = "config.yaml"

# Number of training examples to use
_C.N_TRAIN = None

_C.TRAIN = CN()

_C.TRAIN.IMG_SIZE = 128
_C.TRAIN.LR = 2e-5
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.WEIGHT_DECAY = 1e-2
_C.TRAIN.N_EPOCHS = 1
_C.TRAIN.N_FOLDS = 3
_C.TRAIN.VALID_STEP = 10
_C.TRAIN.ACCU_GRAD_STEPS = 1
_C.TRAIN.GRAD_CLIP = 1.0
_C.TRAIN.MODEL_NAME = "a_model"
_C.TRAIN.NUM_WORKERS = 0
_C.TRAIN.MODEL_SAVE_PATH = "../saved_models/baseline"
# threshold for early stopping (unit is number of epochs)
_C.TRAIN.ES_THRESH = 3


# ---------------------------------------------------------------------------- #
# Path options
# ---------------------------------------------------------------------------- #
_C.PATHS = CN()
# Data Path
_C.PATHS.SHARD_ROOT = ""
# Data file
_C.PATHS.DATAFILE = ""
# Output directory parent folder
_C.PATHS.OUT_DIR = ""
# Results log file name
_C.PATHS.RESULTS_LOG_FILENAME = "results.csv"
# Experiment name
_C.PATHS.EXPERIMENT_NAME = ""
# Experiment description
_C.PATHS.EXPERIMENT_DESC = ""
# Get current timestamp
_C.PATHS.TIMESTAMP = "at_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# Outdirectory for TB logging
_C.PATHS.TB_OUT_DIR = ""
# Outdirectory for model checkpoints
_C.PATHS.MODEL_OUT_DIR = ""
# path to pretrained transformer file (if any)
_C.PATHS.PRETRAINED_TRANSFORMER_FILE = ""


def assert_and_infer_cfg(cache_urls=True):
    """Checks config values invariants."""
    pass


def dump_cfg(cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(cfg.PATHS.OUT_DIR, cfg.CFG_DEST)
    with open(cfg_file, "w") as f:
        cfg.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
