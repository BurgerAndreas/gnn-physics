import os

import random
import pandas as pd
import numpy as np
import time
import tqdm
import copy
import matplotlib.pyplot as plt
import functools
import json
import os
import enum

import torch
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import DataLoader
import torch.optim as optim

import tensorflow
import tensorflow.compat.v1 as tfv1

root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir, "data/datasets/cylinder_flow")
checkpoint_dir = os.path.join(root_dir, "best_models")
postprocess_dir = os.path.join(root_dir, "animations")


import torchdata
from torchdata.datapipes.iter import FileLister, FileOpener, TFRecordLoader


def my_ftrecord_to_pytorch():
    """
    tfrecord is a binary file like H5DF"""
    datapipe1 = FileLister(dataset_dir, "*.tfrecord")
    print(len(list(datapipe1)), ".tfrecord files found in ", dataset_dir)

    datapipe2 = FileOpener(str(dataset_dir) + "train.tfrecord", mode="b")

    with open(os.path.join(dataset_dir, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())

    datapipe2 = FileOpener(datapipe1, mode="b")
    tfrecord_loader_dp = datapipe2.load_from_tfrecord()
    print(type(tfrecord_loader_dp))

    dp = TFRecordLoader(datapipe2, spec=meta)
    dp.map()
    return tfrecord_loader_dp


if __name__ == "__main__":
    my_ftrecord_to_pytorch()
