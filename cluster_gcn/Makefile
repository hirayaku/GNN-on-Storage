CXX=c++
INCFLAGS=$(shell python3 -m pybind11 --includes)
CXXFLAGS=-std=gnu++11 -fopenmp -O3 -shared -fPIC

all: feat_shuffle.so

feat_shuffle.so: feat_shuffle.cpp
	$(CXX) $(INCFLAGS) $^ -o $@ $(CXXFLAGS)

.PHONY: clean

clean:
	rm -f *.so *.o

