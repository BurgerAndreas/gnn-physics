# Notes on usage and todos

#### run
conda activate /home/andreasburger/miniconda3/envs/mlp11
/home/andreasburger/miniconda3/envs/mlp11/bin/python /ssd/Code/gnn-physics/data/datasets/hdf5_to_pyg.py

#### wandb
wandb.init(mode="disabled") or by setting WANDB_MODE=disable

### ToDo

[] animation works
[] lossplot works

[] animate a rollout

[] delete checkpoints, plots
[] rerun

[] pip, conda env
[] install.sh



### Run things

ssh andreasburger@10.70.2.74

SCREEN_NAME='BahenScreen'
screen -d -m $SCREEN_NAME -c "bashrc; ~/Code/gnn-physics/x_run_gnn.sh; exec sh"

screen -ls
screen -r $SCREEN_NAME



---