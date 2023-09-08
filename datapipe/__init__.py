from .base_pipes import IterDataPipe, make_functional
from . import custom_pipes, parallel_pipes

# the following is not necessary
from multiprocessing.reduction import ForkingPickler
import torch.multiprocessing # make sure the tensor reduction methods are registered

class OverridingPickler(ForkingPickler):
    def __init__(self, *args, **kwargs):
        args = list(args)
        args.append(kwargs.get('protocol', None))
        super().__init__(*args)

import io
def _dump(obj, file, protocol=None, *, fix_imports=True, buffer_callback=None):
    OverridingPickler(file, protocol, fix_imports=fix_imports,
             buffer_callback=buffer_callback).dump(obj)

def _dumps(obj, protocol=None, *, fix_imports=True, buffer_callback=None):
    bytes_types = (bytes, bytearray)
    f = io.BytesIO()
    OverridingPickler(f, protocol, fix_imports=fix_imports,
             buffer_callback=buffer_callback).dump(obj)
    res = f.getvalue()
    assert isinstance(res, bytes_types)
    return res

import pickle
_ori_pickles = pickle.Pickler, pickle.dumps, pickle.dump

def override_pickle():
    # override default pickler, is there a better way?
    pickle.Pickler = OverridingPickler
    pickle.dumps = _dumps
    pickle.dump = _dump

def restore_pickle():
    pickle.Pickler, pickle.dumps, pickle.dump = _ori_pickles
