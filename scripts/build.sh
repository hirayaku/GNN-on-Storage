#!/bin/bash

cd torch_io
python3 -m pip install -t . .

TORCH_LIBS=$(python3 -c "import os,torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIBS:$LD_LIBRARY_PATH
stubgen -m gnnos -o .

