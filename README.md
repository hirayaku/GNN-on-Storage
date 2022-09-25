# GNN-on-Storage

Prerequisites: python>=3.6, python modules: `torch, torchmetrics, dgl, tqdm, pyinstrument, mypy, ogb, tensorboard, shutils`

Build `gnnos` module:
```bash
bash scripts/build.sh
```

Experiments: see scripts under `sbatch` folder.

## Commands to limit memory resources

```
sudo bash scripts/set_memory_limit.sh <xxxG> <username> # create a group named "memory:gnnxxxG"

cgexec -g memory:gnnxxxG <your-executable> # run executable within the memory budget
```

