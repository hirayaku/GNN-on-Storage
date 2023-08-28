# NOTE: measure and compare the latency of passing tensors in two ways:
# using a shared memory manager vs. using the default sharing strategy of pytorch
# pytorch is generally 3x~4x slower
import sys, time, os
import torch
import torch.multiprocessing as mp
from data.io import TensorMeta
from data.shmem import ShmemTensor, ShmemClient, ShmemManager, ShmemBackend
from data.shmem_v6d import ShmemBackendV6D

def consumer(backend: ShmemBackend, msg_ch):
    count = 0
    now = time.time()
    while True:
        msg = msg_ch.get()
        if msg[0] is False:
            break
        tinfo = msg[1]
        tensor = ShmemTensor(tinfo, backend)
        print(f"Tensor shared in {time.time()-now:.2f}s")
        print("Recv tensor:", tinfo)
        print(tensor[:8])
        backend.free(tinfo.path)
        count += 1
        now = time.time()

def share_v6d():
    backend = ShmemBackendV6D(cap='2G')
    backend.startup()

    msg_q = mp.Queue(1)
    proc = mp.Process(target=consumer, args=(backend, msg_q), daemon=True)
    proc.start()

    for i in range(16):
        tinfo = TensorMeta([1024*1024*256], torch.float)
        while True:
            try:
                tensor = ShmemTensor(tinfo, backend)
                tensor[:] = i
                break
            except Exception as e:
                print("Failed to create new Tensor:", e)
                time.sleep(0.1)
        msg_q.put((True, tinfo))
        print(f"[{i}] Sent tensor:", tinfo)
    msg_q.put((False,))

    proc.join()
    backend.shutdown()

def consumer2(msg_ch):
    count = 0
    now = time.time()
    while True:
        msg = msg_ch.get()
        if msg[0] is False:
            break
        tensor = msg[1]
        print(f"Tensor shared in {time.time()-now:.2f}s")
        print(tensor[:8])
        count += 1
        now = time.time()

def share_torch():
    msg_q = mp.Queue(1)
    proc = mp.Process(target=consumer2, args=(msg_q,), daemon=True)
    proc.start()

    float_tensor = torch.empty([1]).float()
    for i in range(16):
        size = (1024*1024*256,)
        storage = torch.FloatStorage._new_shared(size[0])
        tensor = float_tensor.new(storage)
        tensor[:] = i
        msg_q.put((True, tensor))
        print(f"[{i}] Sent tensor")
    msg_q.put((False,))
    proc.join()

def fn(client):
    bid, buffer = client.allocate(4)
    print("Client reads:", memoryview(buffer).tobytes().decode())
    print(buffer.call("size"))
    bid, buffer = client.allocate(4)
    print("Client reads:", memoryview(buffer).tobytes().decode())
    bid, buffer = client.allocate(4)
    print("Client reads:", memoryview(buffer).tobytes().decode())

def client():
    # NOTE: there are several methods to manage the client connection

    # 1) using `with` block:
    # __exit__ will be called upon throwing exceptions, closing the connection
    # the allocated buffer won't be completely released
    #
    # with ShmemClient(session) as client:
    #     fn(client)

    # 2) using `try: ... finally: ...`
    # the finally block has the fall-back code that will close the connection
    # same as method 1), the allocated buffer won't be completely released
    #
    # client = ShmemClient(session)
    # try:
    #     fn(client)
    # except:
    #     pass
    # finally:
    #     client.close()

    # 3) no fall-back code
    # client won't be clsoed automatically upon exceptions
    # but all allocated buffers could be released through the alive client connection
    #
    # client = ShmemClient(session)
    # fn(client)
    # client.close()

    client = ShmemClient(session)
    if (os.fork() == 0):
        bid, buffer = client.allocate(4)
        print("Client reads:", memoryview(buffer).tobytes().decode())

def get_size(self):
    return self.size()

from data import SharedBuffer
SharedBuffer.FlatBuffer.register("size", get_size)

if __name__ == "__main__":
    import logging
    logger = logging.getLogger()
    logger.level = logging.DEBUG
    # share_torch()
    # share_v6d()
    session = 'shmem_test'
    if len(sys.argv) > 1:
        cap = 1024 * 1024
        manager = ShmemManager(session, cap=cap)
        try:
            manager.startup(nonblocking=False)
        finally:
            manager.shutdown()
    else:
        # with ShmemClient(session) as client:
        #     bid, buffer = client.allocate(4)
        #     print("Client reads:", buffer)
        client()
