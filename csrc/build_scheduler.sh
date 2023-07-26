TORCH_PATH=/home/cxh/work/anaconda3/lib/python3.8/site-packages/torch

echo "g++ -O3 -fopenmp block_scheduler.cpp -o block_scheduler -I$TORCH_PATH/include/torch/csrc/api/include/ -I$TORCH_PATH/include -L$TORCH_PATH/lib -D_GLIBCXX_USE_CXX11_ABI=0 -lgomp  -ltorch -ltorch_cpu -lc10"

g++ -O3 -fopenmp block_scheduler.cpp -o block_scheduler -I$TORCH_PATH/include/torch/csrc/api/include/ -I$TORCH_PATH/include -L$TORCH_PATH/lib -D_GLIBCXX_USE_CXX11_ABI=0 -lgomp  -ltorch -ltorch_cpu -lc10

#export LD_LIBRARY_PATH=/home/cxh/work/anaconda3/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH

#./block_scheduler
