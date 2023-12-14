# This file is heavily based on
# https://colab.research.google.com/drive/1mZAWP6k9R0DE5NxPzF8yL2HpIUG3aoDC?usp=sharing

import torch
import os
import random
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import math

from torch_geometric.data import Data

# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

import numpy as np
import time
import torch.optim as optim
import tqdm
import pandas as pd
import copy
import copy
import pickle

import gnn.gnn as models
import gnn.utils as stats
import gnn.plotting as plotting
from gnn.dirs import CHECKPOINT_DIR, DATASET_DIR, PLOTS_DIR, DELTA_T


def add_noise(dataset, cfg):
    """Add noise to each timestep.

    noise_field: 'velocity' for cylinder_flow in the original codebase
    noise_scale: 0.02 for cylinder_flow in the original codebase
    noise_gamma: 1.0 for cylinder_flow in the original codebase

    Similar to split_and_preprocess() from
    https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets
    """
    # dataset_before = copy.deepcopy(dataset)
    for datapoint in dataset:
        # each datapoint is class torch_geometric.data.Data
        # add noise to velocity
        momentum = datapoint.x[:, :2]  # (tsteps, 2)
        node_type = datapoint.x[:, 2:]  # (tsteps, 7)
        # noise
        noise = torch.empty(momentum.shape).normal_(mean=0.0, std=cfg.data.noise_scale)
        # but don't apply noise to boundary nodes
        condition = node_type[:, 0] == torch.ones_like(node_type[:, 0])  # (tsteps)
        condition = condition.unsqueeze(1)  # (tsteps, 1)
        condition = condition.repeat(1, 2)  # (tsteps, 2)
        # noise (tsteps, 2)
        noise = torch.where(
            condition=condition, input=noise, other=torch.zeros_like(momentum)
        )
        momentum += noise
        datapoint.x = torch.cat((momentum, node_type), dim=-1).type(torch.float)
        datapoint.y += (1.0 - cfg.data.noise_gamma) * noise  # (tsteps, 2)
        # print('Still the same?', dataset_before[0].x == datapoint.x)
    return dataset


def train(data_train, data_test, stats_list, cfg):
    """
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    """

    # add noise to the training data
    if cfg.data.noise_scale > 0.0:
        data_train = add_noise(data_train, cfg)

    assert (
        len(data_train) > 0 and len(data_test) > 0
    ), f"Start training on {len(data_train)} train and {len(data_test)} test datapoints"

    # torch_geometric DataLoaders are used for handling the data of lists of graphs
    # data is already shuffled if we want it, so do not shuffle again
    loader = DataLoader(
        data_train,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        data_test,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    # The statistics of the data are decomposed
    [
        mean_vec_x,
        std_vec_x,
        mean_vec_edge,
        std_vec_edge,
        mean_vec_y,
        std_vec_y,
    ] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (
        mean_vec_x.to(cfg.device),
        std_vec_x.to(cfg.device),
        mean_vec_edge.to(cfg.device),
        std_vec_edge.to(cfg.device),
        mean_vec_y.to(cfg.device),
        std_vec_y.to(cfg.device),
    )

    # Define the model name for saving checkpoint
    model_name = plotting.name_from_config(cfg)
    path_model_checkpoint = os.path.join(CHECKPOINT_DIR, model_name + "_model.pt")
    path_infos = os.path.join(CHECKPOINT_DIR, model_name + "_infos.pkl")
    path_df = os.path.join(CHECKPOINT_DIR, model_name + "_losses.pkl")
    # saving model
    if not os.path.isdir(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    # look for checkpoint
    # if it exists, continue from previous checkpoint
    if os.path.exists(path_model_checkpoint) and (
        cfg.training.resume_checkpoint == True
    ):
        # get infos
        with open(path_infos, "rb") as f:
            infos = pickle.load(f)
        (num_node_features, num_edge_features, num_classes, stats_list) = infos

        # instantiate model
        model = models.MeshGraphNet(
            num_node_features, num_edge_features, cfg.model.hidden_dim, num_classes, cfg
        ).to(cfg.device)
        model.load_state_dict(
            torch.load(path_model_checkpoint, map_location=cfg.device)
        )
        print("Continuing from previous checkpoint.")

    else:
        # build model
        num_node_features = data_train[0].x.shape[1]
        num_edge_features = data_test[0].edge_attr.shape[1]
        num_classes = 2  # the dynamic variables have the shape of 2 (velocity)

        # save data infos
        infos = (num_node_features, num_edge_features, num_classes, stats_list)

        model = models.MeshGraphNet(
            num_node_features, num_edge_features, cfg.model.hidden_dim, num_classes, cfg
        ).to(cfg.device)

        # dataframe with losses
        df = pd.DataFrame(columns=["epoch", "train_loss", "test_loss", "velocity_val_loss"])

        print("No previous checkpoint found. Starting training from scratch.")

    # Paper used Adam optimizer with no learning rate schedule.
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.opt_restart)

    # train
    losses = []
    test_losses = []
    velocity_val_losses = []
    best_test_loss = np.inf
    best_model = None
    for epoch in tqdm.trange(cfg.training.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops = 0
        for batch in loader:
            # Note that normalization must be done before it's called. The unnormalized
            # data needs to be preserved in order to correctly calculate the loss
            batch = batch.to(cfg.device)
            optimizer.zero_grad()
            pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss = model.loss(pred, batch, mean_vec_y, std_vec_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_loops += 1
        total_loss /= num_loops
        losses.append(total_loss)

        # Every tenth epoch, calculate acceleration test loss (prediction)
        # and velocity validation loss
        if epoch % 10 == 0:
            test_loss, velocity_val_rmse = test(
                test_loader,
                cfg.device,
                model,
                mean_vec_x,
                std_vec_x,
                mean_vec_edge,
                std_vec_edge,
                mean_vec_y,
                std_vec_y,
            )
            velocity_val_losses.append(velocity_val_rmse.item())
            test_losses.append(test_loss.item())

            # save the model if the current one is better than the previous best
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)

        else:
            # If not the tenth epoch, append the previously calculated loss to the
            # list in order to be able to plot it on the same plot as the training losses
            test_losses.append(test_losses[-1])
            velocity_val_losses.append(velocity_val_losses[-1])

        # log to dataframe
        df.loc[len(df.index)] = [epoch, losses[-1], test_losses[-1],  velocity_val_losses[-1]] 

        wandb.log(
            {
                "train_loss": losses[-1],
                "test_loss": test_losses[-1],
                "velocity_loss": velocity_val_losses[-1],
            }
        )

        if epoch % 100 == 0:
            tqdm.tqdm.write(
                "train loss "
                + str(round(total_loss, 2))
                + " | test loss "
                + str(round(test_loss.item(), 2))
                + " | velocity loss "
                + str(round(velocity_val_rmse.item(), 5)),
            )

            if cfg.training.save_best_model:
                # model
                torch.save(best_model.state_dict(), path_model_checkpoint)
                # data infos
                with open(path_infos, "wb") as f:
                    pickle.dump(infos, f)
                # losses
                df.to_pickle(path_df)

    print("Finished training!")
    print("Min test set loss:                {0}".format(min(test_losses)))
    print("Minimum loss:                     {0}".format(min(losses)))
    print("Minimum velocity validation loss: {0}".format(min(velocity_val_losses)))

    if (best_model is not None) and cfg.training.save_best_model:
        # model
        torch.save(best_model.state_dict(), path_model_checkpoint)
        # data infos
        with open(path_infos, "wb") as f:
            pickle.dump(infos, f)
        # losses
        df.to_pickle(path_df)
        print("Saving best model to", str(path_model_checkpoint))

    return


def test(
    loader,
    device,
    test_model,
    mean_vec_x,
    std_vec_x,
    mean_vec_edge,
    std_vec_edge,
    mean_vec_y,
    std_vec_y,
    is_validation=True,
):
    """
    Calculates test set losses and validation set errors.
    """

    loss = 0
    velocity_rmse = 0
    num_loops = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            # calculate the loss for the model given the test set
            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            test_loss = test_model.loss(pred, data, mean_vec_y, std_vec_y)
            loss += test_loss

            # calculate validation error
            # Like for the MeshGraphNets model, 
            # calculate the mask over which we calculate the flow loss 
            # and add this calculated RMSE value to our val error
            normal = torch.tensor(0)
            outflow = torch.tensor(5)
            loss_mask = torch.logical_or(
                (torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(0)),
                (torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(5)),
            )

            eval_velocity = (
                data.x[:, 0:2]
                + stats.unnormalize(pred[:], mean_vec_y, std_vec_y) * DELTA_T
            )
            gs_velocity = data.x[:, 0:2] + data.y[:] * DELTA_T

            error = torch.sum((eval_velocity - gs_velocity) ** 2, axis=1)
            velocity_rmse += torch.sqrt(torch.mean(error[loss_mask]))

        num_loops += 1

    return (loss / num_loops), (velocity_rmse / num_loops)


@hydra.main(version_base=None, config_path="conf", config_name="default")
def load_train_plot(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    wandb.login()
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="MeshGraphNets", name=plotting.name_from_config(cfg))

    print("Cuda is available to torch:", torch.cuda.is_available())
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.set_default_device(cfg.device)

    # Set the random seeds for all random number generators
    torch.manual_seed(cfg.rseed)  # Torch
    random.seed(cfg.rseed)  # Python
    np.random.seed(cfg.rseed)  # NumPy

    # load the data for training and testing
    # there are 4 * 599 timesteps in the provided .tar dataset
    # if you specify more steps, the code will behave in unexpected ways
    file_path = os.path.join(DATASET_DIR, cfg.data.datapath)
    if (cfg.data.train_test_same_traj == True) and (cfg.data.single_traj == True):
        # test on later timesteps of the same trajectory
        # if you specify more than 599 steps, it will still select data from multiple trajectories
        dataset_train = torch.load(file_path)[: cfg.training.train_size]
        dataset_test = torch.load(file_path)[
            cfg.training.train_size : (cfg.training.train_size + cfg.training.test_size)
        ]
    elif cfg.data.train_test_same_traj == True:
        # take random timesteps from the same soup of trajectories
        dataset = torch.load(file_path)
        random.shuffle(dataset)
        dataset_train = dataset[: cfg.training.train_size]
        # test
        dataset_test = dataset[
            cfg.training.train_size : (cfg.training.train_size + cfg.training.test_size)
        ]
    else:
        # test on a different trajectory
        test_file_path = file_path.replace("train", "test")
        dataset_train = torch.load(file_path)[: cfg.training.train_size]
        dataset_test = torch.load(test_file_path)[: cfg.training.test_size]

    # timesteps in random order
    random.shuffle(dataset_train)
    random.shuffle(dataset_test)

    # maybe it would be better to load the full data to compute the statistics
    # this would ensure that we can use the same model checkpoint on different sets of data.
    # currently we have to recompute the statistic for each loaded dataset
    # stats has to happen on the CPU, because the dataset is a list
    stats_list = stats.get_stats(dataset_train + dataset_test)

    # Training
    train(
        data_train=dataset_train, data_test=dataset_test, stats_list=stats_list, cfg=cfg
    )

    f = plotting.save_plots(cfg)
    wandb.log({"figure": f})

    anim_path = plotting.animate_rollout(cfg)
    wandb.log({"animation": anim_path})

    return


if __name__ == "__main__":
    load_train_plot()
