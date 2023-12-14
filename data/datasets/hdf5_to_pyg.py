import os
import pathlib
import numpy as np
import torch
import h5py
from torch_geometric.data import Data
import enum
import argparse

import tensorflow.compat.v1 as tf


class NodeType(enum.IntEnum):
    """
    A 9-dimensional one-hot vector corresponding to node location
    in fluid, wall, inflow, or outflow regions.
    From
    https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/common.py
    """

    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def triangles_to_edges(faces):
    """Computes mesh edges from triangles.
    From
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    # collect edges from triangles
    edges = tf.concat(
        [faces[:, 0:2], faces[:, 1:3], tf.stack([faces[:, 2], faces[:, 0]], axis=1)],
        axis=0,
    )
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers = tf.reduce_min(edges, axis=1)
    senders = tf.reduce_max(edges, axis=1)
    packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
    # remove duplicates and unpack
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    senders, receivers = tf.unstack(unique_edges, axis=1)
    # create two-way connectivity
    return (
        tf.concat([senders, receivers], axis=0),
        tf.concat([receivers, senders], axis=0),
    )


def load_hdf5_save_pt(
    infile: str | pathlib.Path,
    outfile: str | pathlib.Path,
    num_traj: int = -1,
    num_tsteps: int = -1,
) -> None:
    """Load .hdf5, process data, and save as PyTorch Graph format in .pt files.

    Combines all timesteps from all trajectories into one list.
    Adds the next timestep as a target (=ground truth to predict) to each timestep.
    Add noise to each timestep.

    Combines add_targets() and split_and_preprocess() from dataset.py from
    https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets

    infile:
    outfile:
    num_traj: max number of trajectories to use. -1 means use all.
    num_tsteps: max number of timesteps per trajectory to use. -1 means use all.

    Adapted from
    https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing
    and
    https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets
    """
    # Define the list that will return the data graphs
    # all trajectories are 'flattened' (=stacked) into this list
    data_list = []

    # timestep in the simulation which generated the data
    # = timesteps between the graphs
    # from meta.json in the original dataset
    # A constant: do not change!
    dt = 0.01

    # data is saved as a dict[str, dict[str, np.array]]
    # see tfrecord_to_hdf5.py: save_numpy_as_hdf5()
    data = h5py.File(infile, "r")

    num_traj_cnt = 0
    num_tsteps_cnt = 0
    with h5py.File(infile, "r") as data:
        for i, trajectory in enumerate(data.keys()):
            if i == num_traj:
                break
            num_traj_cnt += 1
            num_tsteps_cnt += len(data[trajectory]["velocity"]) - 1

            # We iterate over all the time steps to produce an example graph
            # except for the last one
            # which does not have a 'next time step'
            # to use as a target
            for ts in range(len(data[trajectory]["velocity"]) - 1):
                if ts == num_tsteps:
                    break

                # Get node features
                # Note that it's faster to convert to numpy then to torch than to
                # import to torch from h5 format directly
                momentum = torch.tensor(np.array(data[trajectory]["velocity"][ts]))
                # node_type = torch.tensor(np.array(data[trajectory]['node_type'][ts]))
                node_type = torch.tensor(
                    np.array(
                        tf.one_hot(
                            tf.convert_to_tensor(data[trajectory]["node_type"][0]),
                            NodeType.SIZE,
                        )
                    )
                ).squeeze(1)

                # Get edge indices in COO format
                edges = triangles_to_edges(
                    tf.convert_to_tensor(np.array(data[trajectory]["cells"][ts]))
                )

                edge_index = torch.cat(
                    (
                        torch.tensor(edges[0].numpy()).unsqueeze(0),
                        torch.tensor(edges[1].numpy()).unsqueeze(0),
                    ),
                    dim=0,
                ).type(torch.long)

                # Get edge features
                u_i = torch.tensor(np.array(data[trajectory]["mesh_pos"][ts]))[
                    edge_index[0]
                ]
                u_j = torch.tensor(np.array(data[trajectory]["mesh_pos"][ts]))[
                    edge_index[1]
                ]
                u_ij = u_i - u_j
                u_ij_norm = torch.norm(u_ij, p=2, dim=1, keepdim=True)
                edge_attr = torch.cat((u_ij, u_ij_norm), dim=-1).type(torch.float)

                # Node outputs, targets for training (velocity)
                v_t = torch.tensor(np.array(data[trajectory]["velocity"][ts]))
                v_tp1 = torch.tensor(np.array(data[trajectory]["velocity"][ts + 1]))
                y = ((v_tp1 - v_t) / dt).type(torch.float)

                # Node outputs, for testing integrator (pressure)
                p = torch.tensor(np.array(data[trajectory]["pressure"][ts]))

                # Data needed for visualization code
                cells = torch.tensor(np.array(data[trajectory]["cells"][ts]))
                mesh_pos = torch.tensor(np.array(data[trajectory]["mesh_pos"][ts]))

                # Note that other datasets than cylinder_flow have different feature fields
                # which currently cannot be handle by the code

                # # add noise
                # noise = torch.empty(momentum.shape).normal_(mean=0., std=noise_scale)
                # # but don't apply noise to boundary nodes
                # condition = node_type[:, 0] == torch.ones_like(node_type[:, 0]) # (tsteps)
                # condition = condition.unsqueeze(1) # (tsteps, 1)
                # condition = condition.repeat(1, 2) # (tsteps, 2)
                # # noise (tsteps, 2)
                # noise = torch.where(condition=condition, input=noise, other=torch.zeros_like(momentum))
                # momentum += noise
                # y += (1.0 - noise_gamma) * noise

                x = torch.cat((momentum, node_type), dim=-1).type(torch.float)
                data_list.append(
                    Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y,  # velocity
                        p=p,  # pressure
                        cells=cells,
                        mesh_pos=mesh_pos,
                    )
                )

    outfile = str(outfile) + ".pt"
    torch.save(data_list, outfile)
    print(
        "Saved",
        num_tsteps_cnt,
        "timesteps from",
        num_traj_cnt,
        "trajectories to",
        outfile + " (load_hdf5_save_pt)",
    )
    return


if __name__ == "__main__":
    print("-" * 80, "\nStart", __file__)

    parser = argparse.ArgumentParser()

    # python ./data/datasets/hdf5_to_pyg.py -in 'data/datasets/cylinder_flow_hdf5/train.hdf5' -out 'data/datasets/cylinder_flow_pyg/train.pt' --num_traj 3
    parser.add_argument(
        "-in",
        "--indir",
        dest="indir",
        default="data/datasets/cylinder_flow_hdf5/train",
        help=".hdf5 file to load",
        type=str,
    )
    parser.add_argument(
        "-out",
        "--outdir",
        dest="outdir",
        default="data/datasets/cylinder_flow_pyg/train",
        help=".pt file to output",
        type=str,
    )

    args = parser.parse_args()

    dataset_dir = pathlib.Path(__file__).parent
    data_dir = dataset_dir.parent
    root_dir = data_dir.parent

    # try to find the file
    file_dir = None
    if os.path.exists(f"{root_dir}/{args.indir}"):
        file_dir = root_dir
    elif os.path.exists(f"{data_dir}/{args.indir}"):
        file_dir = data_dir
    elif os.path.exists(f"{dataset_dir}/{args.indir}"):
        file_dir = dataset_dir
    if file_dir is None:
        print("Could not find file", f"{root_dir}/{args.indir}")

    # Define the data folder and data file name
    infile = os.path.join(file_dir, args.indir)
    outfile = os.path.join(file_dir, args.outdir)
    outfile = os.path.splitext(outfile)[0]

    load_hdf5_save_pt(infile=infile, outfile=outfile)

    print("End", __file__)
    print("-" * 80)
