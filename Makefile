# Paths where MAGMA, CUDA, and OpenBLAS are installed.
# MAGMADIR can be .. to test without installing.
#MAGMADIR     ?= ..
MAGMADIR     ?= /usr/local/magma
CUDADIR      ?= /usr/local/cuda
OPENBLASDIR  ?= /usr/local/OpenBLAS
MPICXX ?= mpic++

CC            = gcc
CXX						= g++
FORT          = gfortran
LD            = gcc
CFLAGS        = -Wall
CXXFLAGS			= -O3 -g -rdynamic -std=c++11 -fopenmp
NVCC 					= nvcc
NVCCFLAGS     = -O3 -g -std=c++11
LDFLAGS       = -Wall #-fopenmp
MPICFLAGS = -I/usr/local/openmpi/include -L/usr/local/openmpi/lib -lmpi


# ----------------------------------------
# Flags and paths to MAGMA, CUDA, and LAPACK/BLAS
MAGMA_CFLAGS     := -DADD_ \
                    -I$(MAGMADIR)/include \
                    -I$(CUDADIR)/include


# may be lib instead of lib64 on some systems
MAGMA_LIBS       := -L$(MAGMADIR)/lib -lmagma_sparse -lmagma \
                    -L$(CUDADIR)/lib64 -lcublas -lcudart -lcuda -lcusparse \
                    -L$(OPENBLASDIR)/lib -lopenblas -lpthread

all: main test generate-weight

generate-weight: generate-weight.cpp
	$(CXX) $(CXXFLAGS) $(MAGMA_CFLAGS) -o $@ $^ $(MAGMA_LIBS)

main: main.o helper_lib.o
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(MAGMA_LIBS)

test: test.o helper_lib.o
	$(MPICXX) $(CXXFLAGS) -o $@ $^ $(MAGMA_LIBS)

test.o: test.cpp
	$(CXX) $(CXXFLAGS) $(MAGMA_CFLAGS) $(MPICFLAGS) -c $? -o $@

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(MAGMA_CFLAGS) $(MPICFLAGS) -c $? -o $@

helper_lib.o: helper_lib.cu
	$(NVCC) $(NVCCFLAGS) $(MPICFLAGS) $(MAGMA_CFLAGS) -c $? -o $@

clean:
	-rm -f *.o main test generate-weight
