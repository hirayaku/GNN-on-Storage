import logging
import torch

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

try:
    torch.classes.load_library('data/build/libxTensor.so')
except:
    print("Fail to load xTensor")
    raise

# logger = logging.getLogger()
# try:
#     from torch.utils.cpp_extension import load
#     load(name="SharedBuffer", sources=["data/csrc/flat_buffer.cpp"], extra_cflags=['-O2'])
#     import SharedBuffer
# except:
#     logger.warn("Fail to load module: SharedBuffer")
