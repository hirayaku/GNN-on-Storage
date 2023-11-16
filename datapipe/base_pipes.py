import functools
from typing import Dict, Callable, Optional, TypeVar
from torch.utils.data import IterableDataset

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

class IterData:
    __slots__ = 'val', 'sync'
    def __init__(self, val, sync:bool=False):
        self.val = val
        self.sync = sync

class IterDataPipe(IterableDataset[T_co]):
    '''
    IterDataPipe borrowed from torch datapipes but without hook_iterator
    '''
    functions: Dict[str, Callable] = {}
    reduce_ex_hook: Optional[Callable] = None
    getstate_hook: Optional[Callable] = None
    str_hook: Optional[Callable] = None
    repr_hook: Optional[Callable] = None

    def __getattr__(self, attribute_name):
        if attribute_name in IterDataPipe.functions:
            function = functools.partial(IterDataPipe.functions[attribute_name], self)
            return function
        else:
            raise AttributeError("'{0}' object has no attribute '{1}".format(self.__class__.__name__, attribute_name))

    @classmethod
    def register_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register):
        if function_name in cls.functions:
            raise Exception("Unable to add DataPipe function name {} as it is already taken".format(function_name))

        def class_function(cls, source_dp, *args, **kwargs):
            return cls(source_dp, *args, **kwargs)

        function = functools.partial(class_function, cls_to_register)
        cls.functions[function_name] = function

    def __getstate__(self):
        """
        This contains special logic to serialize `lambda` functions when `dill` is available.
        If this doesn't cover your custom DataPipe's use case, consider writing custom methods for
        `__getstate__` and `__setstate__`, or use `pickle.dumps` for serialization.
        """
        state = self.__dict__
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __reduce_ex__(self, *args, **kwargs):
        if IterDataPipe.reduce_ex_hook is not None:
            try:
                return IterDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn):
        if IterDataPipe.getstate_hook is not None and hook_fn is not None:
            raise Exception("Attempt to override existing getstate_hook")
        IterDataPipe.getstate_hook = hook_fn

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn):
        if IterDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise Exception("Attempt to override existing reduce_ex_hook")
        IterDataPipe.reduce_ex_hook = hook_fn

    def __repr__(self):
        if self.repr_hook is not None:
            return self.repr_hook(self)
        # Instead of showing <torch. ... .MapperIterDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __str__(self):
        if self.str_hook is not None:
            return self.str_hook(self)
        # Instead of showing <torch. ... .MapperIterDataPipe object at 0x.....>, return the class name
        return str(self.__class__.__qualname__)

    def __dir__(self):
        # for auto-completion in a REPL (e.g. Jupyter notebook)
        return list(super().__dir__()) + list(self.functions.keys())

    def __iter__(self):
        raise NotImplementedError

    def reset(self) -> None:
        r"""
        Reset the `IterDataPipe` to the initial state. By default, no-op. For subclasses of `IterDataPipe`,
        depending on their functionalities, they may want to override this method with implementations that
        may clear the buffers and reset pointers of the DataPipe.
        The `reset` method is always called when `__iter__` is called as part of `hook_iterator`.
        """
        pass

class make_functional:
    name: str

    def __init__(self, name: str, enable_df_api_tracing=False) -> None:
        """
            Args:
                enable_df_api_tracing - if set, any returned DataPipe would accept
                DataFrames API in tracing mode.
        """
        self.name = name
        self.enable_df_api_tracing = enable_df_api_tracing

    def __call__(self, cls):
        if issubclass(cls, IterDataPipe):
            IterDataPipe.register_datapipe_as_function(self.name, cls)
        return cls
