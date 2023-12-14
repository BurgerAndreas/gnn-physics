import os
import pathlib

# `path.parents[1]` is the same as `path.parent.parent`
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATASET_DIR = os.path.join(ROOT_DIR, "data/datasets")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "data/checkpoints")
PLOTS_DIR = os.path.join(ROOT_DIR, "data/2d_loss_plots")
ANIM_DIR = os.path.join(ROOT_DIR, "data/animations")

# constant from the original dataset (timestep simulation)
DELTA_T = 0.01
