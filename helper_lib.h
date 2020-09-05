#ifndef HLP_LIB_H
#define HLP_LIB_H

#include <cuda_runtime.h>
#include <mpi.h>
#include <string>


struct Config{
  std::string xFileName, yFileName;
  int row, col, classNum, hiddenNeuron;
  float alpha;
  std::string wInputFileName, wOutputFileName, runningTimeFileName;
};

struct ConfigTest{
  std::string xFileName, yFileName;
  int row, col, classNum, hiddenNeuron;
  double alpha;
  std::string wInputFileName, wOutputFileName, accuracyFileName;
};


struct RunningTimeData{
  int np, row, col, hiddenNeuron;
  double readDataTime, writeDataTime, generateWeightTime, maxH, maxA, maxW, combineW, totalTime, realTotalTime;
};

struct AccuracyData{
  double RMSE, TrueAccuracy;
};

void randomize_smatrix(float *arr, int row, int col);
Config readConfig(char **argv);
/*
  subcount = count of subdata (K)
*/
void getRowSplitSize(int totalRow, int subCount, int subIdx, int *row, int *rowOffset);
void read_smatrix(MPI_Comm comm, std::string fileName, float *arr, int row, int rowOffset, int col);
void transpose_smatrix(float *src, float *dst, int row, int col);
void activationFunction(cudaStream_t cudaStream, float *d_A, int row, int col, int *d_row, int *d_col);
/*
  d_Adiag = d_Adiag (substract?-,+) d_alfa
*/
void addToDiagonal(cudaStream_t cudaStream, float *d_A, int row, int col, int *d_row, int *d_col, float *d_alfa, bool substract);
// ConfigTest readConfigTest(std::string configFileName);
// void writeRunningTimeData(std::string fileName, RunningTimeData rt);
// void writeMatrixfToFileBinary(std::string fileName, float* matrix, int m, int n);
// void writeAccuracyData(std::string fileName, AccuracyData accuracyData);
#endif
