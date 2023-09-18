from typing import Optional, Callable
import time, math, warnings, queue, threading
from functools import partial
import torch
import torch.multiprocessing as mp
from datapipe import make_functional, IterDataPipe
import datapipe.communication as comm
import utils

def worker_init_fn(dp: IterDataPipe, num_par: Optional[int],
                   affinity: Optional[list], init_fn: Optional[Callable],
                   **kwargs):
    if num_par is not None:
        torch.set_num_threads(num_par)
    utils.set_affinity(affinity)
    if init_fn is not None:
        return init_fn(dp, **kwargs)

def make_dp_worker(dp: IterDataPipe, multiprocessing_ctx, worker_name=None,
                   num_par=None, affinity=None, init_fn=None) -> IterDataPipe:
    worker, req_queue, res_queue = comm.eventloop.CreateProcessForDataPipeline(
        multiprocessing_ctx, dp, process_name=worker_name,
        call_on_process_init=partial(worker_init_fn, num_par=num_par, affinity=affinity, init_fn=init_fn)
        # call_on_process_init could in addition 1) apply sharding or 2) connect to prior dp behind queues
    )
    worker.daemon = True
    worker.start()
    proxy_dp = comm.iter.QueueWrapper(comm.protocol.IterDataPipeQueueProtocolClient(req_queue, res_queue))
    return proxy_dp

@make_functional("pmap")
class ParallelMapperDataPipe(IterDataPipe):
    '''
    Spawn `num_par` instances of `fn`, which take items from the shared `source_dp`,
    process the data, and put the results on the shared output queue to be iterated.
    '''
    QUEUE_GET_TIMEOUT = 0.001
    def __init__(self, source_dp: IterDataPipe,
                 fn: Callable, num_par:int=0, num_intra_par:int=1,
                 mp_ctx=None, affinity: Optional[list[int]]=None):
        self.source_dp = source_dp
        self.fn = fn
        # number of workers
        self.num_par = num_par
        # threads allocated for torch operations within each worker
        self.num_intra_par = num_intra_par
        self.mp_ctx = mp_ctx
        self._worker_initialized = False
        self._worker_affinity = affinity
    
    @staticmethod
    def _clear_queue(q: queue.Queue):
        try:
            while True:
                d = q.get_nowait()
                warnings.warn("leftover items found in queue:", d)
        except queue.Empty:
            pass

    def __iter__(self):
        '''
        returns an iterator by initializing the internal states or, *clearing states*
        and reusing the workers
        '''
        self.source_count = 0
        self.target_count = 0
        if self.num_par == 0:
            for data in self.source_dp:
                yield self.fn(data)
                del data
        else:
            # assert isinstance(self.source_dp, comm.iter.QueueWrapper)
            if self.mp_ctx is None:
                self.mp_ctx = mp.get_context()
            if not self._worker_initialized:
                self._worker_stop = self.mp_ctx.Event()
                self._worker_restart = self.mp_ctx.Event()
                self._barrier = self.mp_ctx.Barrier(parties=self.num_par)
                self._worker_initialized = True
                self.source_queue = self.mp_ctx.Queue(maxsize=self.num_par)
                self.target_queue = self.mp_ctx.Queue(maxsize=self.num_par)
                self._create_process_group()
            else:
                # reset states to reuse workers
                self._worker_stop.clear()
                self._worker_restart.set()
            self._create_thread_worker()
            stop_recvd = False
            while True:
                data = self.target_queue.get()
                if isinstance(data, StopIteration):
                    stop_recvd = True
                    # NB: adding the check because the items from queue might be out-of-order
                    # e.g. the StopIteration object, which should come in the last, might actually
                    # preceed several data objects
                    if self.target_count == self.source_count:
                        break
                elif isinstance(data, comm.messages.ExceptionWrapper):
                    data.reraise()
                else:
                    self.target_count += 1
                    yield data
                del data
                if stop_recvd and self.target_count == self.source_count:
                    break
            self._clear_queue(self.target_queue)

    def __len__(self):
        return len(self.source_dp)

    def _thread_worker(self):
        for data in self.source_dp:
            self.source_count += 1
            self.source_queue.put(data)
        self.source_queue.put(StopIteration(f"{self.__class__}: source_queue"))

    def _create_thread_worker(self):
        thread = threading.Thread(target=self._thread_worker, daemon=True)
        thread.start()
    
    def _process_loop_fn(self) -> bool:
        try:
            data = self.source_queue.get(block=True, timeout=self.QUEUE_GET_TIMEOUT)
            if isinstance(data, StopIteration):
                self._worker_stop.set()
                # raise queue.Empty(f"{self.__class__}: StopIteration received by worker")
                return True
            result = self.fn(data)
            self.target_queue.put(result)
            self.target_count += 1
        except queue.Empty:
            # time.sleep(self.QUEUE_GET_TIMEOUT)
            return self._worker_stop.is_set()
        except:
            # something bad happens during fn
            worker_name = mp.current_process().name
            exec_wrapper = comm.messages.ExceptionWrapper(where="in "+worker_name)
            self._worker_stop.set()
            self.target_queue.put(exec_wrapper)
            return True
        else:
            return False

    def _process_worker(self, worker_affinity=None):
        torch.init_num_threads()
        torch.set_num_threads(self.num_intra_par)
        worker_affinity = self._worker_affinity if worker_affinity is None else worker_affinity
        utils.set_affinity(worker_affinity)
        # the main processing loop
        forever = True
        while forever:
            stop = self._process_loop_fn()
            if stop:
                # print(f"processed {self.target_count} items")
                if self._barrier.wait() == 0:
                    # append a StopIteration after all data items
                    self.target_queue.put(StopIteration(f"{self.__class__}: target_queue"))
                    self._barrier.reset()
                self._worker_restart.wait()
                if self._barrier.wait() == 0:
                    # make the restart event usable for the next run
                    self._worker_restart.clear()
                    self._barrier.reset()
                self.target_count = 0

    def _create_process_group(self):
        workers = []
        for i in range(self.num_par):
            w = self.mp_ctx.Process(
                target=self._process_worker, name=f"ParMapper-{i}", daemon=True,
            )
            workers.append(w)
        for w in workers:
            w.start()
        self.workers = workers

@make_functional("pmapv2")
class ParallelMapperV2DataPipe(IterDataPipe):
    '''
    Spawn `num_par` instances of `fn`, which take items from the shared `source_dp`,
    process the data, and put the results on the shared output queue to be iterated.
    '''
    QUEUE_GET_TIMEOUT = 0.01
    QUEUE_FANOUT = 1
    def __init__(self, source_dp: IterDataPipe,
                 fn: Callable, num_par:int=0, num_intra_par:int=1,
                 mp_ctx=None, affinity: Optional[list[int]]=None):
        self.source_dp = source_dp
        self.fn = fn
        # number of workers
        self.num_par = num_par
        # threads allocated for torch operations within each worker
        self.num_intra_par = num_intra_par
        self.mp_ctx = mp_ctx
        self._worker_initialized = False
        self._worker_affinity = affinity
    
    @staticmethod
    def _clear_queue(q: queue.Queue):
        try:
            while True:
                d = q.get_nowait()
                warnings.warn(f"leftover items found in queue: {d}")
        except queue.Empty:
            pass
    
    def __iter__(self):
        '''
        returns an iterator by initializing the internal states or, *clearing states*
        and reusing the workers
        '''
        if self.num_par == 0:
            for data in self.source_dp:
                yield self.fn(data)
        else:
            num_queues = math.ceil(self.num_par / self.QUEUE_FANOUT)
            self.num_queues = num_queues
            self.source_count = [0] * num_queues
            self.target_count = [0] * num_queues
            # assert isinstance(self.source_dp, comm.iter.QueueWrapper)
            if self.mp_ctx is None:
                self.mp_ctx = mp.get_context()
            if not self._worker_initialized:
                self._worker_stop = [self.mp_ctx.Event() for _ in range(num_queues)]
                self._worker_restart = self.mp_ctx.Event()
                self._barrier = self.mp_ctx.Barrier(parties=self.num_par)
                self._worker_initialized = True
                self.source_queues = [self.mp_ctx.Queue(maxsize=self.QUEUE_FANOUT*2)
                                      for _ in range(num_queues)]
                self.target_queues = [self.mp_ctx.Queue(maxsize=self.QUEUE_FANOUT*2)
                                      for _ in range(num_queues)]
                self._create_process_group()
            else:
                # reset states to reuse workers
                for i in range(num_queues):
                    self._worker_stop[i].clear()
                self._worker_restart.set()
            self._create_thread_worker()

            # poll items from target queues
            # XXX incomplete implementation
            i = 0
            while True:
                data = self.target_queues[i].get()
                if isinstance(data, StopIteration):
                    break
                elif isinstance(data, comm.messages.ExceptionWrapper):
                    data.reraise()
                else:
                    yield data
                i = (i + 1) % self.num_queues
            for target_queue in self.target_queues:
                self._clear_queue(target_queue)
    
    def __len__(self):
        return len(self.source_dp)
    
    def _thread_worker(self):
        i = 0
        for data in self.source_dp:
            self.source_queues[i].put(data)
            self.source_count[i] += 1
            i = (i + 1) % self.num_queues
        for source_queue in self.source_queues:
            source_queue.put(StopIteration(f"{self.__class__}: source_queue"))
    
    def _create_thread_worker(self):
        thread = threading.Thread(target=self._thread_worker, daemon=True)
        thread.start()
    
    def _process_loop_fn(self, source_queue, target_queue, _worker_stop) -> bool:
        try:
            data = source_queue.get(block=True, timeout=self.QUEUE_GET_TIMEOUT)
            if isinstance(data, StopIteration):
                _worker_stop.set()
                # raise queue.Empty(f"{self.__class__}: StopIteration received by worker")
                return True
            result = self.fn(data)
            target_queue.put(result)
        except queue.Empty:
            # time.sleep(self.QUEUE_GET_TIMEOUT)
            return _worker_stop.is_set()
        except:
            # something bad happens during fn
            worker_name = mp.current_process().name
            exec_wrapper = comm.messages.ExceptionWrapper(where="in "+worker_name)
            _worker_stop.set()
            target_queue.put(exec_wrapper)
            return True
        else:
            return False

    def _process_worker(self, i:int, worker_affinity=None):
        torch.init_num_threads()
        torch.set_num_threads(self.num_intra_par)
        worker_affinity = self._worker_affinity if worker_affinity is None else worker_affinity
        utils.set_affinity(worker_affinity)

        qid = i // self.QUEUE_FANOUT
        source_queue, target_queue = self.source_queues[qid], self.target_queues[qid]
        _worker_stop = self._worker_stop[qid]
        forever = True
        while forever:
            stop = self._process_loop_fn(source_queue, target_queue, _worker_stop)
            if stop:
                # print(f"processed {self.target_count} items")
                if self._barrier.wait() == 0:
                    # append a StopIteration after all data items
                    target_queue.put(StopIteration(f"{self.__class__}: target_queue"))
                    self._barrier.reset()
                self._worker_restart.wait()
                if self._barrier.wait() == 0:
                    # make the restart event usable for the next run
                    self._worker_restart.clear()
                    self._barrier.reset()
                self.target_count = 0

    def _create_process_group(self):
        workers = []
        for i in range(self.num_par):
            w = self.mp_ctx.Process(
                target=self._process_worker, args=(i,),
                name=f"ParMapper-{i}", daemon=True,
            )
            workers.append(w)
        for w in workers:
            w.start()
        self.workers = workers
