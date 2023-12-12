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
import h5py

import torch
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import DataLoader
import torch.optim as optim

import tensorflow as tf
import tensorflow.compat.v1 as tfv1

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
# for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done

root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir, "data/datasets/cylinder_flow")
checkpoint_dir = os.path.join(root_dir, "best_models")
postprocess_dir = os.path.join(root_dir, "animations")


class NodeType(enum.IntEnum):
    """
    a 9-dimensional one-hot vector corresponding to node
    location in fluid, wall, inflow, or outflow regions.
    From https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/common.py
    """

    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def _parse(proto, meta):
    """Parses a trajectory from tf.Example, i.e. decode (map) the data.
    A tf.Example proto stores the data format, i.e. data shape and type.
    A tf.Example contains key-value Example.features where each key (string)
    maps to a tf.train.Feature message which contains a fixed-type list.
    """
    feature_lists = {k: tfv1.io.VarLenFeature(tfv1.string) for k in meta["field_names"]}
    features = tfv1.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tfv1.io.decode_raw(features[key].values, getattr(tfv1, field["dtype"]))
        data = tfv1.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tfv1.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tfv1.io.decode_raw(features["length_" + key].values, tfv1.int32)
            length = tfv1.reshape(length, [-1])
            data = tfv1.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset and decode (map) it.
    From https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/dataset.py
    path
    split: train, test, valid
    """
    # meta.json contains infos about the data format saved in .tfrecord
    # cells (time, cell, three nodes): describes the mesh in this trajectory.
    # mesh_pos (time, node, two coord): 2d coordinates of the nodes
    with open(os.path.join(path, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())
    ds = tfv1.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds


def add_targets(ds, fields, add_history, traj_len=-1):
    """Adds target and optionally history fields to dataframe.
    CylinderFlow does not use history in the paper.
    Adds the next timestep as the 'target'.
    Removes first timestep (because there is no history)
    and last timestep (because it has no next timestep 'target')
    from all data fields.
    """

    def fn(trajectory):
        out = {}
        for key, val in trajectory.items():
            out[key] = val[1:-1]
            if key in fields:
                if add_history:
                    out["prev|" + key] = val[0:-2]
                out["target|" + key] = val[2:]
            # trim to only contain traj_len timesteps
            if traj_len == -1:
                pass
            elif traj_len < len(out[key]):
                out[key] = out[key][:traj_len]
                if key in fields:
                    if add_history:
                        out["prev|" + key] = out["prev|" + key][:traj_len]
                    out["target|" + key] = out["target|" + key][:traj_len]
        return out

    return ds.map(fn, num_parallel_calls=8)


def split_and_preprocess(
    ds, noise_field, noise_scale, noise_gamma, shuffle=True, num_prefetch=3
):
    """Splits trajectories into frames, and adds training noise."""

    def add_noise(frame):
        noise = tfv1.random.normal(
            tfv1.shape(frame[noise_field]), stddev=noise_scale, dtype=tfv1.float32
        )
        # don't apply noise to boundary nodes
        mask = tfv1.equal(frame["node_type"], NodeType.NORMAL)[:, 0]
        noise = tfv1.where(mask, noise, tfv1.zeros_like(noise))
        frame[noise_field] += noise
        frame["target|" + noise_field] += (1.0 - noise_gamma) * noise
        return frame

    # flatten the data i.e. stack all trajectories into one sequence
    ds = ds.flat_map(tfv1.data.Dataset.from_tensor_slices)
    ds = ds.map(add_noise, num_parallel_calls=4)
    # randomly samples timesteps
    # will always resample, i.e. we can draw steps forever
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    # repeat dataset. None: indefinitely times
    ds = ds.repeat(count=None)
    # Prefetch: While the model is executing training step s
    # the input pipeline is reading the data for step s+1.
    return ds.prefetch(num_prefetch)


def batch_dataset(ds, batch_size):
    """Batches input datasets."""
    shapes = ds.output_shapes
    types = ds.output_types

    def renumber(buffer, frame):
        nodes, cells = buffer
        new_nodes, new_cells = frame
        return nodes + new_nodes, tfv1.concat([cells, new_cells + nodes], axis=0)

    def batch_accumulate(ds_window):
        out = {}
        for key, ds_val in ds_window.items():
            initial = tfv1.zeros((0, shapes[key][1]), dtype=types[key])
            if key == "cells":
                # renumber node indices in cells
                num_nodes = ds_window["node_type"].map(lambda x: tfv1.shape(x)[0])
                cells = tfv1.data.Dataset.zip((num_nodes, ds_val))
                initial = (tfv1.constant(0, tfv1.int32), initial)
                _, out[key] = cells.reduce(initial, renumber)
            else:
                merge = lambda prev, cur: tfv1.concat([prev, cur], axis=0)
                out[key] = ds_val.reduce(initial, merge)
        return out

    if batch_size > 1:
        ds = ds.window(batch_size, drop_remainder=True)
        ds = ds.map(batch_accumulate, num_parallel_calls=8)
    return ds




if __name__ == "__main__":
    """
    From run_model.py
    For CylinderFlow (cfd)
    """
    print("-" * 80, "\nIn", __file__)

    params = dict(
        # how much noise to add to the input
        noise=0.02,
        # how much noise to add to the target to compensate for the noise added to the input
        # gamma=1.0 adds no noise to the target, gamma=0 adds the same amount as to the input
        gamma=1.0,
        # the quantity which is predicted by the GNN
        field="velocity",
        #
        history=False,
        #
        size=2,
        # batch size
        batch=2,
    )

    num_trajectories = 3
    tstep_train = 50
    tstep_test = 25
    tsteps = tstep_train + tstep_test

    ds = load_dataset(dataset_dir, "train")
    ds = ds.take(num_trajectories)
    ds_numpy = ds_to_numpy(ds)
    save_numpy_as_hdf5(ds_numpy, fname='./data/cylinder_flow_hdf5/test')

    ds_loaded = load_hdf5(fname='./data/cylinder_flow_hdf5/test')

    # for elem in ds.take(1):
    #     print(type(elem))
    #     print(elem['cells'].shape)
    #     print(elem['cells'][0])

    ds = add_targets(
        ds, [params["field"]], add_history=params["history"], traj_len=tsteps
    )

    # print("num_trajectories:", num_trajectories)
    # tsteps_target = 0
    # i = 0
    # for elem in ds:
    #     print("trajectory", i, "has", ds.element_spec["cells"].shape[0], "timesteps")
    #     tsteps_target += ds.element_spec["cells"].shape[0]
    #     i += 1
    # print("Total of ", tsteps_target, "timesteps")

    # # 'cells': <tf.Tensor: shape=(598, 3518, 3)
    # # (timesteps, cells, nodes)
    # # loop over timesteps
    # for elem in ds.take(num_trajectories):
    #     print(type(elem))
    #     print(elem['cells'].shape)
    #     print(elem['cells'][0])

    # it = iter(ds)
    # print(next(it).numpy())

    ds = split_and_preprocess(
        ds,
        noise_field=params["field"],
        noise_scale=params["noise"],
        noise_gamma=params["gamma"],
        shuffle=False,
    )
    # print("\nsplit_and_preprocess")
    # print(ds.element_spec)

    # iterating over a dataset requires eager execution

    # iterate, batches

    # iterator = iter(ds)

    # batch_size = 2
    # ds = ds.batch(batch_size)
    # iterator = ds.make_initializable_iterator()
    # batch_data = iterator.get_next()

    # # Shuffle
    # dataset  = dataset.shuffle(buffer_size=1e5)
    # # Specify batch size
    # dataset  = dataset.batch(128)
    # # Calculate and print batch size
    # batch_size = next(iter(dataset)).shape[0]
    # print('Batch size:', batch_size) # prints 128

    # for key in inputs.keys():
    #     print(key, inputs[key].numpy().shape)
    # cells (3374, 3)
    # mesh_pos (1804, 2)
    # node_type (1804, 1)
    # velocity (1804, 2)
    # target|velocity (1804, 2)
    # pressure (1804, 1)

    # dataset = tfv1.data.Dataset.from_tensor_slices([1, 2, 3])
    # for element in dataset.as_numpy_iterator():
    #   print(element)

    """
    “Encoding” generates node and edge embeddings from features, 
    “processing” takes care of message passing, aggregation, and updating, 
    decoding is the post-processing step that gives final predictions
    """
