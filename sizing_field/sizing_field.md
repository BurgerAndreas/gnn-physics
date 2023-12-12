# Sizing field

The [original codebase](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) 
does not contain the prediction of the sizing field and the corresponding remesher.

Only the `flag_dynamic_sizing` (36 GB) and `sphere_dynamic_sizing` datasets can be used to learn the sizing field.
```bash
bash ./data/datasets/download_dataset.sh flag_dynamic_sizing ./data/datasets
```

