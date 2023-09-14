import os, gc, tempfile, warnings, array, pickle, socket, threading
from dataclasses import fields
from typing import Tuple, Optional
from enum import Enum, auto
import torch
from data.io import TensorMeta
from data import SharedBuffer
import logging
logger = logging.getLogger()

# NOTE: When relying on torch to pass tensor around,
# we have to wait a bit before exit so that the tensors could be safely shared
# (assuming the tensor sharing strategy is using fd).
# Otherwise, the fd in the current process will be closed early before it gets passed
# via socket and used by the consumer process.

def fds_from_ancdata(ancdata):
    fds = array.array("i")   # Array of ints
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            # Append data, ignoring any truncated integers at the end.
            fds.frombytes(cmsg_data[:len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
    return list(fds)

def send_msg(sock: socket.socket, bufs, fds: list[int]) -> int:
    if fds is not None and len(fds) > 0:
        ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))]
    else:
        ancdata = []
    return sock.sendmsg(bufs, ancdata)

def recv_msg(sock: socket.socket, msglen: int, maxfds: int) -> Tuple[bytes, list]:
    fds = array.array("i")   # Array of ints
    msg_buf, ancdata, _, _ = sock.recvmsg(msglen, socket.CMSG_LEN(maxfds * fds.itemsize))
    return msg_buf, fds_from_ancdata(ancdata)

MAX_LISTEN_BACKLOG = 32
MSG_BUF_SIZE = 1024
MAX_FDS = 2

ShmemID = str
class ShmemBuffer(object):
    '''
    A ShmemBuffer object is associated with a ShmemClient
    When the ShmemBuffer gets deleted, it should notify the client to release the memory
    '''
    def __init__(self, bid: ShmemID, addr: int, size: int): ...
    def __del__(self): ...

def _serialize(buffer: SharedBuffer.FlatBuffer):
    client = buffer.client()
    client.inc(buffer.id())
    return buffer.session(), buffer.id()

def _deserialize(state):
    session, bid = state
    client = get_client(session)
    return client.get(bid)

class ShmemMsgType(Enum):
    GREETINGS   = auto()    # res: msg=[type, size:int], anc=[fd:int]
    REQUEST_MEM = auto()    # req: msg=[type, size:int, align: int]
                            # res: RESPOND_MEM  | REFRESH_MEM
    REFER_MEM   = auto()    # req: msg=[type, bid: str]
                            # res: RESPOND_MEM | REFRESH_MEM
    RELEASE_MEM = auto()    # req: msg=[type, bid: str]
    FAREWELL    = auto()    # req: msg=[type]
    RESPOND_MEM = auto()    # res: msg=[type, ok:bool, bid: str, offset:int, size:int]
    REFRESH_MEM = auto()    # res: msg=[type, ok:bool, bid: str, offset:int, size:int],anc=[fd:int]

class ShmemClient(object):
    class ClientException(Exception):
        pass

    def __init__(self, name):
        self.session = name
        self.sock_name = f"{name}.sock"
        self.sock_file = os.path.join(tempfile.gettempdir(), self.sock_name)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.sock_file)
        self.sock = sock
        self.memfd = None
        self.memsize = 0
        self._closed = False
        # the first message from server contains memfd
        self.greetings()

    def send(self, msg, fds=None):
        buf = pickle.dumps(msg)
        sent = send_msg(self.sock, [buf], fds=fds)
        if sent > MSG_BUF_SIZE:
            warnings.warn(f"ShmemClient message size is too large: {sent}")
 
    def send_recv(self, msg, fds=None):
        self.send(msg, fds)
        return recv_msg(self.sock, MSG_BUF_SIZE, MAX_FDS)

    def handler(self, buf, fds=None) -> Tuple[ShmemID, ShmemBuffer]:
        msg = pickle.loads(buf)
        if not isinstance(msg, list):
            raise TypeError(f"Expects the client message to be a list: {msg}")

        msg_type = msg[0]
        msg_data = msg[1:]
        if msg_type is ShmemMsgType.GREETINGS:
            memfd = fds[0]
            memsize = msg_data[0]
            return self._map_buffer(memfd, memsize)
        elif msg_type is ShmemMsgType.RESPOND_MEM:
            ok, bid, offset, size = msg_data[:4]
            if not ok:
                raise self.ClientException(f"Fail to get memory (id={bid}, size={size})")
            buffer = self.root_buffer.slice(offset, size, bid)
            return bid, buffer
        # elif msg_type is ShmemMsgType.REFRESH_MEM: ...
        else:
            raise ValueError(f"Unknown message type from manager: {msg}")
 
    def _map_buffer(self, memfd, size):
        os.lseek(memfd, 0, os.SEEK_SET)
        stat = os.fstat(memfd)
        logging.info(f"Client receives fd={memfd}, size={stat.st_size}")
        self.root_buffer = SharedBuffer.FlatBuffer(memfd, size, self)
        self.memfd = memfd
        self.memsize = size
 
    def greetings(self):
        msg = [ShmemMsgType.GREETINGS]
        msg_buf, fds = self.send_recv(msg)
        return self.handler(msg_buf, fds)

    def allocate(self, nbytes: int, align: int = 0):
        '''
        manager allocates memory, returns id & offset in the shared memory region
        client wraps the memory area into a buffer object to be used later,
        and ask the backend to increment the refcount of `id'. When the buffer object
        gets freed, the refcount of `id` will be decremented.
        '''
        msg = [ShmemMsgType.REQUEST_MEM, nbytes, align]
        msg_buf, fds = self.send_recv(msg)
        return self.handler(msg_buf, fds)
    
    def inc(self, bid: ShmemID):
        '''
        Increment reference to the shared memory region.
        '''
        msg = [ShmemMsgType.REFER_MEM, bid]
        self.send_recv(msg)

    def dec(self, bid: ShmemID):
        self.free(bid)

    def get(self, bid: ShmemID):
        '''
        Get a reference to the shared memory region.
        Refcount of id is incremented in the manager side.
        '''
        msg = [ShmemMsgType.REFER_MEM, bid]
        msg_buf, fds = self.send_recv(msg)
        return self.handler(msg_buf, fds)

    def free(self, bid: ShmemID) -> None:
        msg = [ShmemMsgType.RELEASE_MEM, bid]
        buf = pickle.dumps(msg)
        send_msg(self.sock, [buf], fds=None)

    def close(self):
        if not self._closed:
            msg = [ShmemMsgType.FAREWELL]
            buf = pickle.dumps(msg)
            send_msg(self.sock, [buf], fds=None)
            self.sock.close()
            self.root_buffer = None
            self._closed = True
 
    def __enter__(self):
        return self
 
    def __exit__(self, except_t, except_v, traceback):
        self.close()

    # Python doc: __del__ is not guaranteed to be called even when python exits!
    # def __del__(self):
    #     self.close()
    #     super().__del__()

def reduce_buffer(buffer: SharedBuffer.FlatBuffer) -> Tuple:
    return (buffer.session, buffer.id)

def rebuild_client(session):
    return get_shmem_client(session)
 
class ShmemManager(object):
    '''
    Shared memory pool manager which should run in a background process
    '''
    def __init__(self, name: str, cap: int):
        self.sock_name = f"{name}.sock"
        self.sock_file = os.path.join(tempfile.gettempdir(), self.sock_name)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.sock_file)
        self.sock = sock

        # TODO: self.managed_mem = ...
        shm_path = '/dev/shm/pool'
        fd = os.open(shm_path, os.O_RDWR | os.O_CREAT, mode=0o644)
        self.cap = os.write(fd, b'Python rules all!\n')
        # self.cap = cap
        self.fd = fd

    def client_handler(self, client_sock: socket.socket):
        with client_sock:
            while True:
                msg_buf, _ = recv_msg(client_sock, MSG_BUF_SIZE, MAX_FDS)
                msg = pickle.loads(msg_buf)
                print(msg)
                if not isinstance(msg, list):
                    raise TypeError(f"Expects the message from client to be a list: {msg}")
                msg_type = msg[0]
                msg_data = msg[1:]
                if msg_type is ShmemMsgType.GREETINGS:
                    new_msg, fds = [ShmemMsgType.GREETINGS, self.cap], [self.fd]
                    send_msg(client_sock, [pickle.dumps(new_msg)], fds)
                elif msg_type is ShmemMsgType.REQUEST_MEM:
                    # TODO
                    new_msg, fds = [ShmemMsgType.RESPOND_MEM, True, "mem", 0, self.cap], []
                    send_msg(client_sock, [pickle.dumps(new_msg)], fds)
                elif msg_type is ShmemMsgType.RELEASE_MEM:
                    bid = msg_data[0]
                    logging.info(f"Release buffer: id={bid}")
                elif msg_type is ShmemMsgType.FAREWELL:
                    logging.info(f"Disconnect client")
                    break
                else:
                    raise ValueError(f"Unknown message type from client: {msg}")
        # TODO: release all resources allocated to this client when exitting

    def _serve(self):
        self.sock.listen(MAX_LISTEN_BACKLOG)
        logging.info(f"Manager starts listening socket({self.sock.fileno()})")
        while True:
            session_sock, _ = self.sock.accept()
            logging.info("Manager accepts new client connection")
            worker = threading.Thread(target=self.client_handler, args=(session_sock,), daemon=True)
            worker.start()

    def _stop(self):
        self.sock.close()
        # self.sock.shutdown(socket.SHUT_RDWR)
    
    def startup(self, nonblocking=True):
        self.nonblocking = nonblocking
        if nonblocking:
            serve_thread = threading.Thread(target=self._serve, daemon=True)
            serve_thread.start()
            self.serve_thread = serve_thread
        else:
            self._serve()
    
    def shutdown(self):
        self._stop()
        if self.nonblocking:
            self.serve_thread.join()
        os.unlink(self.sock_file)

    def __enter__(self):
        self.startup()
        return self

    def __exit__(self, except_t, except_v, traceback):
        self.shutdown()

class ShmemBackend(object):
    def __init__(self, cap, path): ...
    @property
    def id(self) -> str: ...
    def startup(self) -> None: ...
    def shutdown(self) -> None: ...
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def allocate(self, nbytes: int) -> Tuple[str, memoryview]: ...
    def get(self, bid: str) -> Optional[memoryview]: ...
    def free(self, bid: str) -> None: ...

def ShmemTensor(tinfo: TensorMeta, backend: ShmemBackend, ctx=None, **kwargs) -> torch.Tensor:
    '''
    Create a torch.Tensor backed by a shared memory object store [vineyard]
    '''
    for f in fields(tinfo):
        if f in kwargs:
            setattr(tinfo, f, kwargs[f])
    if tinfo.path is None:
        with backend.allocate(tinfo.nbytes(), ctx) as (obj_id, buffer):
            tensor = torch.frombuffer(buffer, dtype=tinfo.dtype.to_torch())
            tinfo.path = obj_id
    else:
        buffer = backend.get(tinfo.path, ctx)
        tensor = torch.frombuffer(buffer, dtype=tinfo.dtype.to_torch())
    tensor._meta = tinfo.clone()
    return tensor
