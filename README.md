# GNN-on-Storage

Prerequisites: python>=3.8, python modules: `torch, torchmetrics, torch_geometric, torch_sparse, tqdm, json5, pyinstrument, ogb`

## Commands to limit memory resources

```
sudo bash scripts/set_memory_limit.sh <xxxG> <username> # create a group named "memory:gnnxxxG"

cgexec -g memory:gnnxxxG <your-executable> # run executable within the memory budget
```

## Performance notes

Python is generally faster than what you might think (more like ~10x slower instead of 100x slower than C/C++ when running simple loops).
However, there are some cases where you should avoid doing it in Python, and would otherwise cause the ~100x slowdown.

One example is handling millions of items with Python built-in data structures, e.g. `tuple, list, set`:

```python
logger.info(f"Gathering edge parts: {edge_intervals.size(1)}")
edge_parts =  list(zip(
    ranges_gather(src, edge_intervals[0], edge_part_sizes),
    ranges_gather(dst, edge_intervals[0], edge_part_sizes)
)) # slow
edge_parts = [
    (src[start : end], dst[start : end]) for start, end in edge_intervals.t().tolist()
] # slow
logger.info("Gathering completes")
```

Torch datapipes are composable and promising, yet it failed. One reason I suspect the cause of failure: it's really SLOW.

It's slow due to the excessive decorating code. The core concept of torchdata is simple: datapipe graphs with each stage expressed as a python iterator. Note that iterating over a minimal iterator in python is in no case slow: for example, traversing a million-item list takes only 5\~10 ns per iteration. Adopting the generator pattern brings flexibility in implementing datapipe stages but at a cost, generally leading to a 5\~10x slowdown. Yet, up till now, we can still tolerate it somehow.

It's the additional things that pytorch does that makes things intolerable. PyTorch datapipes by default has a lot of decorative code revolving around the core `send`/`yield` loop, which make things **~100x slower**. This overhead applies to all datapipe stages, which really sinks the datapipe performance at runtime.
