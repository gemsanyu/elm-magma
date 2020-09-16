# Paths where MAGMA, CUDA, and OpenBLAS are installed.
# MAGMADIR can be .. to test without installing.
#MAGMADIR     ?= ..
CXX						= g++
CXXFLAGS			= -O3 -std=c++11 -march=native

all: preprocess

preprocess: preprocess.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	-rm -f *.o preprocess
