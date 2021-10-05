#!/usr/bin/bash

CUSTOM_DGL=$(realpath "$(find ../dgl/python -type d -name dgl*.egg)")
export PYTHONPATH="$CUSTOM_DGL:$PYTHONPATH"

