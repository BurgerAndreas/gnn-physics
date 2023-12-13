import os
import pathlib

# `path.parents[1]` is the same as `path.parent.parent`
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
DATASET_DIR = os.path.join(ROOT_DIR, "data/datasets")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "data/checkpoints")
PLOTS_DIR = os.path.join(ROOT_DIR, "data/animations")
