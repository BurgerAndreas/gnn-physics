import os
import pathlib
import numpy as np
import functools
import json
import h5py
import argparse

import tensorflow.compat.v1 as tfv1


def _parse(proto, meta):
    """Parses a trajectory from tf.Example, i.e. decode (map) the data.
    A tf.Example proto stores the data format, like data shape and type.
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


def load_dataset(path: str, split: str):
    """Load dataset and decode (map) it.
    From https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/dataset.py
    path: path to the directory, not including the filename
    split: train, test, valid
    """
    # meta.json contains infos about the data format saved in .tfrecord
    # cells (time, cell, three nodes): describes the mesh in this trajectory.
    # mesh_pos (time, node, two coord): 2d coordinates of the nodes
    # node_type: onehot vector describing were the node is (e.g. wall, outflow)
    # pressure:
    # velocity:
    # note datasets other than cylinder_flow have different feature fields
    with open(os.path.join(path, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())
    ds = tfv1.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
    # decode the data
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    # always prefetch one batch of data to the GPU
    # so we do not have to wait for data and keep the GPU busy
    ds = ds.prefetch(1)
    return ds


def ds_to_numpy(ds) -> dict[str, dict[str, np.array]]:
    """
    output: ds[str(traj_num)]['cells'] -> numpy array
    """
    return {str(num): traj for num, traj in enumerate(ds.as_numpy_iterator())}


def save_numpy_as_hdf5(
    data_dict: dict[str, dict[str, np.array]], fname: str | pathlib.Path
) -> None:
    """
    Write a dictionary of numpy arrays to an hdf5 file.
    https://gist.github.com/gilliss/7e1d451b42441e77ae33a8b790eeeb73
    """
    dir_fname = pathlib.Path(fname).parent
    if not os.path.isdir(dir_fname):
        os.makedirs(dir_fname)
    # open the file to store the data in
    f = h5py.File(fname + ".hdf5", "w")
    # used to store metadata, i.e. format of our numpy arrays
    meta = {}
    for grp_name in data_dict:  # trajectories
        meta[grp_name] = {}
        grp = f.create_group(str(grp_name))
        for dset_name in data_dict[grp_name]:  # cells, mesh_pos, ...
            meta[grp_name][dset_name] = (
                str(type(data_dict[grp_name][dset_name]))
                + " "
                + str(data_dict[grp_name][dset_name].dtype)
            )
            dset = grp.create_dataset(dset_name, data=data_dict[grp_name][dset_name])
    f.close()
    # save metadata
    with open(fname + ".json", "w") as f:
        json.dump(obj=meta, fp=f, indent=4)
    print("Saved", len(data_dict.keys()), "to", fname + ".hdf5 (save_numpy_as_hdf5)")
    return


def load_hdf5(fname):
    with open(fname + ".json") as f:
        meta = json.loads(f.read())
    f = h5py.File(fname + ".hdf5", "r")
    data_dict = {}
    for grp_name in meta:
        data_dict[grp_name] = {}
        for dset_name in meta[grp_name]:
            # to access arrays: f[grp_name][dset_name][:]
            # to access scalars: f[grp_name][dset_name][()]
            data_dict[grp_name][dset_name] = f[grp_name][dset_name][:]
    f.close()
    print("Loaded data from", fname + ".hdf5")
    return data_dict


if __name__ == "__main__":
    print("-" * 80, "\nStart", __file__)

    parser = argparse.ArgumentParser()

    # python ./data/datasets/tfrecord_to_hdf5.py -in 'data/datasets/cylinder_flow/train.tfrecord' -out 'data/datasets/cylinder_flow_hdf5/train.hdf5' --num_traj 3
    parser.add_argument(
        "-in",
        "--indir",
        dest="indir",
        default="data/datasets/cylinder_flow/train",
        help=".tfrecord file to load",
        type=str,
    )
    parser.add_argument(
        "-out",
        "--outdir",
        dest="outdir",
        default="data/datasets/cylinder_flow_hdf5/train",
        help=".hdf5 file to output",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--num_traj",
        dest="num_traj",
        default=1,
        help="How many trajectories in the .tfrecord to convert to .hdf5. -1 means convert all.",
        type=int,
    )

    args = parser.parse_args()

    dataset_dir = pathlib.Path(__file__).parent
    data_dir = dataset_dir.parent
    root_dir = data_dir.parent

    # strip extenstion
    indir = os.path.splitext(args.indir)[0]

    # try to find the file
    if os.path.exists(f"{root_dir}/{indir}.tfrecord"):
        file_dir = root_dir
    elif os.path.exists(f"{data_dir}/{indir}.tfrecord"):
        file_dir = data_dir
    elif os.path.exists(f"{dataset_dir}/{indir}.tfrecord"):
        file_dir = dataset_dir
    else:
        print("Could not find file", f"{root_dir}/{indir}.tfrecord")

    folder_in = pathlib.Path(indir).parent

    # load all trajectories
    ds = load_dataset(f"{file_dir}/{folder_in}", pathlib.Path(indir).stem)
    # only take as many trajectories as we want
    if args.num_traj > 0:
        ds = ds.take(args.num_traj)
    # convert dataset to dict of numpy arrays
    ds_numpy = ds_to_numpy(ds)
    # save dict to numpy arrays as hdf5
    save_numpy_as_hdf5(ds_numpy, fname=f"{file_dir}/{args.outdir}")

    # load the data
    # ds_loaded = load_hdf5(fname=f"{file_dir}/{args.outdir}")

    print("End", __file__)
    print("-" * 80)
