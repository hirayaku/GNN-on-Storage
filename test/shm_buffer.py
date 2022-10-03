import sys, time
import torch
import torch.multiprocessing as mp

from pyinstrument import Profiler

# 16e8 * 4 = 6.4GB
STEP=4000
SIZES=(STEP*10, STEP*10)

def compute(tensor: torch.Tensor) -> torch.Tensor:
    rows = torch.randint(0, tensor.shape[0], (tensor.shape[0]//10,))
    return tensor[rows].sqrt().sum()

def worker_shm(shared_tensor: torch.Tensor, notifyQ: mp.Queue, ackQ: mp.Queue):
    i = 0
    profiler = Profiler()
    while True:
        tok = notifyQ.get()
        if tok == -1:
            return

        profiler.start()
        tic = time.time()
        tensor = shared_tensor[i*STEP:(i+1)*STEP]
        print(f"No.{i} get: {time.time()-tic:.2f}s, shared: {tensor.is_shared()}")

        tic = time.time()
        compute(tensor)
        print(f"No.{i} compute: {time.time()-tic:.2f}s")

        profiler.stop()
        profiler.print()
        ackQ.put(0)
        i += 1

def worker_queue(dataQ: mp.Queue, notifyQ: mp.Queue, ackQ: mp.Queue):
    i = 0
    profiler = Profiler()
    while True:
        tok = notifyQ.get()
        if tok == -1:
            return

        profiler.start()
        tic = time.time()
        tensor = dataQ.get()
        print(f"No.{i} get: {time.time()-tic:.2f}s, shared: {tensor.is_shared()}")

        tic = time.time()
        tensor = tensor[i*STEP:(i+1)*STEP]
        compute(tensor)
        print(f"No.{i} compute: {time.time()-tic:.2f}s")

        profiler.stop()
        profiler.print()
        ackQ.put(0)
        i += 1

if __name__ == "__main__":
    torch.set_num_threads(32)

    buffer = torch.empty(SIZES, dtype=torch.float)
    buffer.share_memory_()

    context = mp.get_context('spawn')
    dataQ = context.Queue(maxsize=1)
    notifyQ = context.Queue(maxsize=1)
    ackQ = context.Queue(maxsize=1)
    ackQ.put(0)

    start = time.time()

    #  using fixed shared tensor to pass data
    if len(sys.argv) > 1 and sys.argv[1] == 's':
        print("share data by reusing shared tensor")
        worker_proc = context.Process(target=worker_shm, args=(buffer, notifyQ, ackQ))
        worker_proc.start()
        for i in range(10):
            tic = time.time()
            data = torch.rand((STEP,STEP*10))
            #  buffer[i*STEP:(i+1)*STEP] = data
            #  torch.rand((STEP,STEP*10), out=buffer[i*STEP:(i+1)*STEP])
            print(f"No.{i} rand: {time.time()-tic:.2f}s")
            ackQ.get()
            notifyQ.put(0)

    else:
        print("share data by sending over queue")
        worker_proc = context.Process(target=worker_queue, args=(dataQ, notifyQ, ackQ))
        worker_proc.start()
        for i in range(10):
            tic = time.time()
            data = torch.zeros((STEP*10,STEP*10))
            torch.rand((STEP,STEP*10), out=data[i*STEP:(i+1)*STEP])
            print(f"No.{i} rand: {time.time()-tic:.2f}s")
            tic = time.time()
            data.share_memory_()
            dataQ.put(data)
            print(f"No.{i} put: {time.time()-tic:.2f}s")
            ackQ.get()
            notifyQ.put(0)

    notifyQ.put(-1)
    worker_proc.join()

    end = time.time()
    print(f"Overall: {end-start:.2f}s")

