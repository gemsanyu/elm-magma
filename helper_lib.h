#ifndef HLP_LIB_H
#define HLP_LIB_H

#include <cuda_runtime.h>
#include <mpi.h>
#include <string>

#include "magma_v2.h"


struct Config{
  std::string xFileName, yFileName;
  int row, col, classNum, hiddenNeuron, subCount;
  float alpha;
  std::string wInputFileName, wOutputFileName, runningTimeFileName;
};

struct ConfigTest{
  std::string xFileName, yFileName;
  int row, col, classNum, hiddenNeuron, subCount;
  double alpha;
  std::string wInputFileName, wOutputFileName, accuracyFileName;
};


struct RunningTimeData{
  int np, subCount, row, col, hiddenNeuron;
  double readDataTime, maxH, maxA, maxW, combineW, totalTime, memoryAllocation;
};

struct AccuracyData{
  double RMSE, TrueAccuracy;
};

void randomize_smatrix(float *arr, int row, int col);
Config readConfig(char **argv);
ConfigTest readConfigTest(char **argv);
/*
  subcount = count of subdata (K)
*/
void getRowSplitSize(int totalRow, int subCount, int subIdx, int *row, int *rowOffset);
void read_smatrix(MPI_Comm comm, std::string fileName, float *d_arr,
  int row, int rowOffset, int col, bool transpose);
void transpose_smatrix(float *src, float *dst, int row, int col);
void activationFunction(cudaStream_t cudaStream, float *d_A, int row, int col, int *d_row, int *d_col);
/*
  d_Adiag = d_Adiag (substract?-,+) d_alfa
*/
void addToDiagonal(cudaStream_t cudaStream, float *d_A, int row, int col, int *d_row, int *d_col, float *d_alfa);
void getPseudoInverse(magma_queue_t queue, float *d_A, float *d_Ainv, int row, int col);
// ConfigTest readConfigTest(std::string configFileName);
void writeRunningTimeData(std::string fileName, RunningTimeData rt);
void write_smatrix(std::string fileName, float* matrix, int m, int n);
// void writeAccuracyData(std::string fileName, AccuracyData accuracyData);
#endif
