import numpy as np
import torch as th

def to_torch_dtype(datatype):
    numpy_to_torch_dtype_dict = {
        np.bool       : th.bool,
        np.uint8      : th.uint8,
        np.int8       : th.int8,
        np.int16      : th.int16,
        np.int32      : th.int32,
        np.int64      : th.int64,
        np.float16    : th.float16,
        np.float32    : th.float32,
        np.float64    : th.float64,
        np.complex64  : th.complex64,
        np.complex128 : th.complex128
    }
    if isinstance(datatype, np.dtype):
        for key in numpy_to_torch_dtype_dict:
            if key == datatype:
                return numpy_to_torch_dtype_dict[key]
    elif isinstance(datatype, th.dtype):
        return datatype

    raise Exception(f'Datatype is neither numpy or torch dtype: {datatype}')

def to_torch_tensor(data):
    if isinstance(data, th.Tensor):
        return data
    else:
        return th.tensor(data)
