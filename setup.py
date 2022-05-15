import os, sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from torch.__config__ import parallel_info

def list_src(dir):
    sources = filter(
        lambda file: "main.cpp" not in file and file.endswith(".cpp"),
        os.listdir(dir))
    return list(map(lambda file: os.path.join(dir, file), sources))

setup(name="gnnos",
    ext_modules=[CppExtension(
        'gnnos',
        list_src("./csrc"),
        extra_compile_args={'cxx': ['-std=c++14', '-g', '-fopenmp']})],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    })
