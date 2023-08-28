import time, gc
import torch
import torch.multiprocessing as torch_mp
from multiprocessing.reduction import DupFd

def fn(q):
    tensor = q.get()
    print(tensor, tensor.is_shared())
    tensor += 1 # process silently crashes
    print("OK")

def test():
    ctx = torch_mp.get_context('spawn')
    q = ctx.Queue()
    tensor = torch.ones(1024*1024)
    tensor.share_memory_()
    proc = ctx.Process(target=fn, args=(q,))
    proc.start()
    q.put(tensor)
    proc.join()

def source_fn(q):
    for _ in range(2):
        t = torch.randint(100, (4,))
        print("source:", t)
        q.put(t)
        del t
        gc.collect()
    time.sleep(1)

def drain_fn(q):
    for _ in range(2):
        t = q.get()
        print("drain:", t)

def fork_only():
    ctx = torch_mp.get_context('fork')
    fork_q = ctx.Queue()
    source = ctx.Process(target=source_fn, args=(fork_q,), daemon=True)
    source.start()
    drain = ctx.Process(target=drain_fn, args=(fork_q,), daemon=True)
    drain.start()
    source.join()
    drain.join()

# NB: you can mix up fork ctx and spawn ctx, as along as you can separte
# direction interactions between fork/spawn data structures
def mix_up():
    ctx = torch_mp.get_context('spawn')
    spwn_q = ctx.Queue()
    source = ctx.Process(target=source_fn, args=(spwn_q,), daemon=True)
    source.start()
    ctx = torch_mp.get_context('fork')
    fork_q = ctx.Queue()
    drain = ctx.Process(target=drain_fn, args=(fork_q,), daemon=True)
    drain.start()

    for _ in range(2):
        fork_q.put(spwn_q.get())
    source.join()
    drain.join()

if __name__ == "__main__":
    fork_only()
    mix_up()