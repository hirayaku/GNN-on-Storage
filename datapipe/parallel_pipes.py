from typing import Optional, Callable
import time, math, warnings, queue, threading
from collections import deque
from functools import partial
import torch
import torch.multiprocessing as mp
from datapipe import make_functional, IterDataPipe
import datapipe.communication as comm
import utils

def worker_init_fn(dp: IterDataPipe, num_par: Optional[int],
                   affinity: Optional[list[int]], init_fn: Optional[Callable],
                   **kwargs):
    torch.init_num_threads()
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
    def __init__(self, source_dp: IterDataPipe, fn: Callable,
                 num_par:int=0, num_intra_par:int=1, num_prefetch:int=0,
                 mp_ctx=None, affinity: Optional[list[int]]=None):
        self.source_dp = source_dp
        self.fn = fn
        # number of workers
        self.num_par = num_par
        self.num_prefetch = num_par if num_prefetch == 0 else num_prefetch
        # threads allocated for torch operations within each worker
        self.num_intra_par = num_intra_par
        self.mp_ctx = mp_ctx
        self.workers = []
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
                data = None
        else:
            # assert isinstance(self.source_dp, comm.iter.QueueWrapper)
            if self.mp_ctx is None:
                self.mp_ctx = mp.get_context()
            if not self._worker_initialized:
                self._worker_stop = self.mp_ctx.Event()
                self._worker_restart = self.mp_ctx.Event()
                self._barrier = self.mp_ctx.Barrier(parties=self.num_par)
                self._worker_initialized = True
                self.source_queue = self.mp_ctx.Queue(maxsize=self.num_prefetch)
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
                data = None
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
            result = self.fn(data); data = None
            self.target_queue.put(result); result = None
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

    def reset(self):
        self.source_dp.reset()
        for w in self.workers:
            w.terminate()
        if len(self.workers) > 0:
            self._clear_queue(self.source_queue)
            self._clear_queue(self.target_queue)

@make_functional("prefetch")
class PrefetcherDataPipe(IterDataPipe):
    '''
    Thread-based datapipe prefetcher. Good for IO concurrency.
    For compute concurrency, consider `make_dp_worker` or `ParallelMapDataPipe`.
    '''
    def __init__(self, source_dp:IterDataPipe, fn:Optional[Callable]=None, buffer_size:int=2):
        self.source_dp = source_dp
        self.fn = fn
        self.buffer_size = buffer_size
        self._buffer = queue.Queue(maxsize=buffer_size)
        self.prefetcher = None

    @staticmethod
    def _clear_queue(q: queue.Queue):
        try:
            while True:
                d = q.get_nowait()
                warnings.warn("leftover items found in queue:", d)
        except queue.Empty:
            pass

    def _thread_prefetcher(self):
        if self.fn is None:
            for data in self.source_dp:
                self._buffer.put(data); data = None
        else:
            for data in self.source_dp:
                self._buffer.put(self.fn(data)); data = None
        self._buffer.put(StopIteration(f"{self.__class__}: prefetcher thread"))

    def __iter__(self):
        if self.buffer_size == 0:
            yield from self.source_dp
        else:
            self._clear_queue(self._buffer)
            self.prefetcher = threading.Thread(target=self._thread_prefetcher, daemon=True)
            self.prefetcher.start()
            while True:
                processed = self._buffer.get()
                if isinstance(processed, StopIteration):
                    break
                else:
                    yield processed
                processed = None

    def reset(self):
        self.source_dp.reset()
        if self.prefetcher is not None:
            self.prefetcher.join()
        self._clear_queue(self._buffer)

@make_functional("prefetch_cuda")
class CudaPrefetcherDataPipe(IterDataPipe):
    '''
    CUDA prefetcher that overlaps data transfer with successive CUDA operations
    * `fn` should be non-blocking calls (like CUDA ops)
    '''
    def __init__(self, source_dp: IterDataPipe, fn: Optional[Callable]=None):
        self.source_dp = source_dp
        self.fn = fn
        self.stream = torch.cuda.Stream()

    def _preload(self):
        try:
            self.next_batch = next(self._source_iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = self.next_batch.cuda(non_blocking=True)
            if self.fn is not None:
                self.fn(self.next_batch)

    def __iter__(self):
        self._source_iter = iter(self.source_dp)
        self._preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            batch.record_stream(torch.cuda.current_stream())
        else:
            raise StopIteration(f"{self.__class__}: source datapipe depleted")
        self._preload()
        return batch

    def reset(self):
        self.source_dp.reset()
