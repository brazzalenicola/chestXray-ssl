import os, sys
import csv
import argparse
import collections
import math
import random
import pandas as pd
import numpy as np
import torch
from torch.autograd.function import InplaceFunction
import scipy.stats as stats
import pytorch_lightning as pl

import transformers
import copy
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn import model_selection

import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D

from core.config import assert_and_infer_cfg, dump_cfg
from core.config import get_cfg_defaults


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="See src/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed)


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def create_folds(data, num_splits, seed):
    data["kfold"] = -1
    kfold = model_selection.KFold(n_splits=num_splits, shuffle=True, random_state=seed)
    for f, (train_idx, valid_idx) in enumerate(kfold.split(X=data)):
        data.loc[valid_idx, "kfold"] = f
    return data


def loss_fn(outputs, targets):
    outputs = outputs.view(-1)
    targets = targets.view(-1)
    return torch.nn.functional.mse_loss(outputs, targets)


def get_metrics(preds, targets):
    preds = preds.detach().cpu().numpy().ravel()
    targets = targets.detach().cpu().numpy().ravel()
    spearman_corr, _ = stats.spearmanr(targets, preds)

    return spearman_corr


def fetch_best_model_filename(model_save_path):
    checkpoint_files = os.listdir(model_save_path)
    best_checkpoint_files = [f for f in checkpoint_files if "best_" in f]
    best_checkpoint_val_loss = [
        float(".".join(x.split("=")[1].split(".")[0:2])) for x in best_checkpoint_files
    ]
    best_idx = np.array(best_checkpoint_val_loss).argmax()
    return os.path.join(model_save_path, best_checkpoint_files[best_idx])


def handle_config_and_log_paths(args):
    # Load default config options
    cfg = get_cfg_defaults()
    # merge config modifications from config file
    cfg.merge_from_file(args.cfg_file)
    # merge config modifications from command line arguments
    cfg.merge_from_list(args.opts)
    # checks and assertions on config
    assert_and_infer_cfg()
    cfg.PATHS.OUT_DIR = os.path.join(
        cfg.PATHS.OUT_DIR, cfg.PATHS.EXPERIMENT_NAME, cfg.PATHS.TIMESTAMP
    )
    model_save_path = os.path.join("../saved_models/", "experiments/")
    # model_save_path = os.path.join(cfg.PATHS.DATAPATH, "experiments/")
    cfg.PATHS.MODEL_OUT_DIR = os.path.join(
        model_save_path, cfg.PATHS.EXPERIMENT_NAME, cfg.PATHS.TIMESTAMP, "saved_models"
    )
    cfg.PATHS.TB_OUT_DIR = os.path.join(cfg.PATHS.OUT_DIR, "tb_logs")

    # freeze config before running experiments
    cfg.freeze()

    # Ensure that the output dir exists
    try:
        os.makedirs(cfg.PATHS.OUT_DIR, exist_ok=True)
        os.makedirs(cfg.PATHS.MODEL_OUT_DIR, exist_ok=True)
        os.makedirs(cfg.PATHS.TB_OUT_DIR, exist_ok=False)
    except FileExistsError:
        print("Wait for a minute and try again :)")
        exit()

    dump_cfg(cfg)

    return cfg


def log_test_results_to_csv(cfg, file_path, test_metrics_l):
    for i, test_metrics_d in enumerate(test_metrics_l):
        test_metrics_d.update(
            {
                "base_lr": cfg.TRAIN.LR,
                "batch_size": cfg.TRAIN.BATCH_SIZE,
                "max_seq_length": cfg.TRAIN.MAX_LEN,
                "max_epochs": cfg.TRAIN.EPOCHS,
                "exp_dir": cfg.PATHS.OUT_DIR,
                "exp_name": cfg.PATHS.EXPERIMENT_NAME,
                "test_id": i,
            }
        )
        log_file = Path(file_path)
        if not log_file.is_file():
            with open(file_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=test_metrics_d.keys())
                writer.writeheader()
                writer.writerow(test_metrics_d)
        else:
            with open(file_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=test_metrics_d.keys())
                writer.writerow(test_metrics_d)


# https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gaussian_nll_loss
def gaussian_nll_loss(input, target, var, *, full=False, eps=1e-6, reduction="mean"):
    r"""Gaussian negative log likelihood loss.
    See :class:`~torch.nn.GaussianNLLLoss` for details.
    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full: ``True``/``False`` (bool), include the constant term in the loss
            calculation. Default: ``False``.
        eps: value added to var, for stability. Default: 1e-6.
        reduction: specifies the reduction to apply to the output:
            `'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    # Inputs and targets much have same shape
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of var must match that of input or be equal to 1
    var = var.view(input.size(0), -1)
    if var.size(1) != input.size(1) and var.size(1) != 1:
        raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate loss (without constant)
    loss = 0.5 * (torch.log(var) + (input - target) ** 2 / var).view(
        input.size(0), -1
    ).sum(dim=1)

    # Add constant to loss term if required
    if full:
        D = input.size(1)
        loss = loss + 0.5 * D * math.log(2 * math.pi)

    # Apply reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class PriorWD(torch.optim.Optimizer):
    def __init__(self, optim, use_prior_wd=False, exclude_last_group=True):
        super(PriorWD, self).__init__(optim.param_groups, optim.defaults)
        self.param_groups = optim.param_groups
        self.optim = optim
        self.use_prior_wd = use_prior_wd
        self.exclude_last_group = exclude_last_group
        self.weight_decay_by_group = []
        for i, group in enumerate(self.param_groups):
            self.weight_decay_by_group.append(group["weight_decay"])
            group["weight_decay"] = 0

        self.prior_params = {}
        for i, group in enumerate(self.param_groups):
            for p in group["params"]:
                self.prior_params[id(p)] = p.detach().clone()

    def step(self, closure=None):
        if self.use_prior_wd:
            for i, group in enumerate(self.param_groups):
                for p in group["params"]:
                    if self.exclude_last_group and i == len(self.param_groups):
                        p.data.add_(
                            -group["lr"] * self.weight_decay_by_group[i], p.data
                        )
                    else:
                        p.data.add_(
                            -group["lr"] * self.weight_decay_by_group[i],
                            p.data - self.prior_params[id(p)],
                        )
        loss = self.optim.step(closure)

        return loss

    def compute_distance_to_prior(self, param):
        assert id(param) in self.prior_params, "parameter not in PriorWD optimizer"
        return (param.data - self.prior_params[id(param)]).pow(2).sum().sqrt()


class Mixout(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError(
                "A mix probability of mixout has to be between 0 and 1,"
                " but got {}".format(p)
            )
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training

        if ctx.p == 0 or not ctx.training:
            return input

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target
        else:
            output = (
                (1 - ctx.noise) * target + ctx.noise * output - ctx.p * target
            ) / (1 - ctx.p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout.apply(input, target, p, training, inplace)
