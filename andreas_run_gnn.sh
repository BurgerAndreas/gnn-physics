#!/bin/bash

# chmod +x run_stanford_simple.sh
# ./run_stanford_simple.sh

cd ~/Code/gnn-physics
# conda activate /home/andreasburger/miniconda3/envs/mlp11
source /home/andreasburger/miniconda3/bin/activate /home/andreasburger/miniconda3/envs/mlp11
# python run_gnn.py
python run_gnn.py +dataset=stanford
python run_gnn.py +noise=paper
# # hypothesis: noise helps when there is more data
# python run_gnn.py +noise=paper +datasize=small
# # more data better
python run_gnn.py +datasize=small
python run_gnn.py +datasize=medium
# # generalize to multiple trajectories
python run_gnn.py +datasize=medium data.single_traj=False
python run_gnn.py +testset=different +datasize=medium
# python run_gnn.py +noise=paper +testset=different +datasize=small

python animate.py
