1. Update the submodule pointing to the customized version of dgl: `git submodule update --init --recursive`
2. Build the customized dgl following the steps [here](https://docs.dgl.ai/install/index.html#install-from-source). In the last step of installation, provide the prefix: `setup.py`: `python setup.py install --prefix .`
3. cd into `./graphsage`, source `env.sh` to prioritize the customized version of dgl over the system default
4. Run the provided training scripts
