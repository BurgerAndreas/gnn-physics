# This file is heavily based on
# https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing

import torch
import os
import random
import pandas as pd
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import DataLoader

import numpy as np
import time
import torch.optim as optim
import tqdm
import pandas as pd
import copy
import matplotlib.pyplot as plt
import h5py
import tensorflow.compat.v1 as tf
import functools
import json
from torch_geometric.data import Data
import enum

from matplotlib import tri as mtri
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


def normalize(to_normalize, mean_vec, std_vec):
    return (to_normalize - mean_vec) / std_vec


def unnormalize(to_unnormalize, mean_vec, std_vec):
    return to_unnormalize * std_vec + mean_vec


def get_stats(data_list):
    """
    Method for normalizing processed datasets. Given  the processed data_list,
    calculates the mean and standard deviation for the node features, edge features,
    and node outputs, and normalizes these using the calculated statistics.
    """

    # mean and std of the node features are calculated
    mean_vec_x = torch.zeros(data_list[0].x.shape[1:])
    std_vec_x = torch.zeros(data_list[0].x.shape[1:])

    # mean and std of the edge features are calculated
    mean_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge = torch.zeros(data_list[0].edge_attr.shape[1:])

    # mean and std of the output parameters are calculated
    mean_vec_y = torch.zeros(data_list[0].y.shape[1:])
    std_vec_y = torch.zeros(data_list[0].y.shape[1:])

    # Define the maximum number of accumulations to perform such that we do
    # not encounter memory issues
    max_accumulations = 10**6

    # Define a very small value for normalizing to
    eps = torch.tensor(1e-8)

    # Define counters used in normalization
    num_accs_x = 0
    num_accs_edge = 0
    num_accs_y = 0

    # Iterate through the data in the list to accumulate statistics
    for dp in data_list:
        # Add to the
        mean_vec_x += torch.sum(dp.x, dim=0)
        std_vec_x += torch.sum(dp.x**2, dim=0)
        num_accs_x += dp.x.shape[0]

        mean_vec_edge += torch.sum(dp.edge_attr, dim=0)
        std_vec_edge += torch.sum(dp.edge_attr**2, dim=0)
        num_accs_edge += dp.edge_attr.shape[0]

        mean_vec_y += torch.sum(dp.y, dim=0)
        std_vec_y += torch.sum(dp.y**2, dim=0)
        num_accs_y += dp.y.shape[0]

        if (
            num_accs_x > max_accumulations
            or num_accs_edge > max_accumulations
            or num_accs_y > max_accumulations
        ):
            break

    mean_vec_x = mean_vec_x / num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x / num_accs_x - mean_vec_x**2), eps)

    mean_vec_edge = mean_vec_edge / num_accs_edge
    std_vec_edge = torch.maximum(
        torch.sqrt(std_vec_edge / num_accs_edge - mean_vec_edge**2), eps
    )

    mean_vec_y = mean_vec_y / num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y / num_accs_y - mean_vec_y**2), eps)

    mean_std_list = [
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ]

    return mean_std_list
