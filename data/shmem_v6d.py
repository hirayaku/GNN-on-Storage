from contextlib import contextmanager
import torch.multiprocessing as mp
from data.shmem import ShmemBackend
import vineyard as v6d

class ShmemBackendV6D(ShmemBackend):
    V6D_SOCK = 'vineyard.sock'
    def __init__(self, cap, sock=V6D_SOCK, mp_ctx=None):
        self.cap = cap
        self.sock = sock
        if mp_ctx is None:
            mp_ctx = mp
        self.notify = mp_ctx.SimpleQueue()
        self.ack = mp_ctx.SimpleQueue()
        self.proc = mp_ctx.Process(target=self.fn)
        self.proc.start()
    def _startup(self, *args):
        v6d.init(size=self.cap, socket=self.sock)
    def _shutdown(self, *args):
        v6d.shutdown()
    def fn(self):
        while True:
            token = self.notify.get()[0]
            if token == 'startup':
                self._startup()
                self.ack.put(None)
            else:
                self._shutdown()
                self.ack.put(None)
                break
    def startup(self):
        self.notify.put(('startup',))
        self.ack.get()
    def shutdown(self):
        self.notify.put(('shutdown',))
        self.ack.get()
        self.proc.join()
    def connect(self) -> v6d.IPCClient:
        return v6d.connect(self.sock)
    def close(self, ctx: v6d.IPCClient):
        ctx.close()
    def get_context(self) -> v6d.IPCClient:
        try:
            return v6d.get_current_client()
        except:
            return self.connect()
    @contextmanager
    def allocate(self, nbytes: int, ctx: v6d.IPCClient=None):
        ctx = self.get_context() if ctx is None else ctx
        blob = None
        try:
            blob = ctx.create_blob(nbytes)
            yield blob.id, blob.buffer
        finally:
            if blob is not None:
                blob.seal(ctx)
    def get(self, ID: str, ctx: v6d.IPCClient=None):
        ctx = self.get_context() if ctx is None else ctx
        blob = ctx.get_blob(v6d.ObjectID(ID))
        return blob.buffer
    def free(self, ID: str, ctx: v6d.IPCClient=None):
        ctx = self.get_context() if ctx is None else ctx
        ctx.delete(v6d.ObjectID(ID))
