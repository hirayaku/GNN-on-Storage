1. Update the submodule pointing to the customized version of dgl: `git submodule update --init --recursive`
2. Build the customized dgl provided by Zheng Da inside `./custom-dgl` following the steps [here](https://docs.dgl.ai/install/index.html#install-from-source). In the last step of installation, provide the prefix: `setup.py`: `python setup.py install --prefix .`
3. Build the official dgl inside `./dgl` following the same steps as 2. This dgl doesn't include the new sampling method
4. cd into `./graphsage`, source `env.sh` to change between the customized version of dgl over the official one
5. Run the provided training scripts
