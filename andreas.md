# Notes on usage and todos

#### run
conda activate /home/andreasburger/miniconda3/envs/mlp11
/home/andreasburger/miniconda3/envs/mlp11/bin/python /ssd/Code/gnn-physics/data/datasets/hdf5_to_pyg.py

#### wandb
wandb.init(mode="disabled") or by setting WANDB_MODE=disable

### ToDo

python animate.py +datasize=medium
python animate.py +datasize=medium +noise=paper
python plot.py +datasize=medium
python plot.py +datasize=medium +noise=paper

[ ] select final animations
[ ] add to gitingore
[ ] update git
[x] write Dave

scp -r andreasburger@10.70.2.74:/ssd/Code/gnn-physics/data/animations /Users/andreasburger/Code




git rm -r --cached .git
git add .
git commit -m 'removed unnecessary files'




### Run things

```bash
ssh andreasburger@10.70.2.74

# detach and kill
screen -X detach BahenScreen
# Ctrl + a, Ctrl + d
screen -S BahenScreen -p 0 -X quit

# start and attach
screen -S BahenScreen
screen -d -m BahenScreen -c "bashrc; /ssd/Code/gnn-physics/andreas_run_gnn.sh; exec sh"

screen -ls
screen -r BahenScreen
```




---