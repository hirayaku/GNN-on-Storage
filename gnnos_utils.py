import torch, gnnos

def store(path, shape, dtype, offset=0):
    '''
    return a new TensorStore given the four parameters
    '''
    return gnnos.tensor_store(
        gnnos.options(path).with_shape(shape).with_dtype(dtype).with_offset(offset)
    )
