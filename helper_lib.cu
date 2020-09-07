#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include "helper_lib.h"

#define BLOCK_SIZE 256
#define GRID_SIZE 1000

using namespace std;


void randomize_smatrix(float *arr, int row, int col){
  unsigned int seed = 42;
  #pragma omp parallel for private(seed)
	for(unsigned int i=0; i<row*col; i++){
    seed += i;
		arr[i]= static_cast <float> (rand_r(&seed)) / static_cast <float> (RAND_MAX);
	}
}

Config readConfig(char **argv){
  Config config;
  string dataName = argv[1];
  config.xFileName = "data/"+dataName+"/training/file_x.bin";
  config.yFileName = "data/"+dataName+"/training/file_y.bin";
  config.row = atoi(argv[2]);
  config.col = atoi(argv[3]);
  config.classNum = atoi(argv[4]);
  config.hiddenNeuron = atoi(argv[5]);
  config.alpha = atoi(argv[6]);
  config.wInputFileName = "weight/"+dataName+"/w-in-"+
    to_string(config.hiddenNeuron);
  config.wOutputFileName = "weight/"+dataName+"/w-out-"+
  to_string(config.hiddenNeuron);
  config.runningTimeFileName = argv[7];
  config.runningTimeFileName = "report/"+dataName+"/"+config.runningTimeFileName;

  cout << "Using " << dataName <<"\n";
  cout << "Total Rows : " << config.row <<"; Cols : "<< config.col<<"; ";
  cout << "Class : " << config.classNum<<"\n";
  cout << "Hidden Neuron : " << config.hiddenNeuron <<"\n";
  return config;
}

void getRowSplitSize(int totalRow, int subCount, int subIdx, int *row, int *rowOffset){

  int rowSplitSizeRemainder = totalRow % subCount;
  int rowCount = totalRow / subCount;
  *row = rowCount;
  if (subIdx < rowSplitSizeRemainder){
    *row += 1;
  }

  // Read Offset for the MPI-PO, offset is sum of rows from n=0, n=rank-1
  // Still needs to be multiplied by size of double and number of column
  // col for x, and classNum for y
  *rowOffset = 0;
  for(int subIter=0; subIter<subIdx; subIter++){
    *rowOffset += rowCount;
    if (subIter < rowSplitSizeRemainder){
      *rowOffset += 1;
    }
  }
}

void read_smatrix(MPI_Comm comm, std::string fileName, float *arr, int row, int rowOffset, int col){
  MPI_Status status;
  MPI_File file;
  float *arrt = (float*) malloc(row*col*sizeof(float));
  char *fileNameChar = const_cast<char*>(fileName.c_str());
  MPI_File_open(comm, fileNameChar, MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
  MPI_File_read_at(file, rowOffset*col*sizeof(float), arrt, row*col, MPI_FLOAT, &status);
  transpose_smatrix(arrt, arr, row, col);
  // free(arrt);
  MPI_File_close(&file);
}

void transpose_smatrix(float *src, float *dst, int row, int col){
  #pragma omp parallel for
  for(int n = 0; n<row*col; n++) {
    int i = n/row;
    int j = n%row;
    dst[n] = src[j*col + i];
  }
}

__global__ void d_ActivationFunction(float *d_A, int *d_row, int *d_col){
  int m = (*d_row);
  int n = (*d_col);
  int size = m*n;
  int stride = blockDim.x*gridDim.x;
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  for(int i=id;i<size;i+=stride){
    d_A[i] = 1.0 / (1.0 + exp(-d_A[i]));
  }
}

void activationFunction(cudaStream_t cudaStream, float *d_A, int row, int col, int *d_row, int *d_col){
  int gridSize = (row*col/BLOCK_SIZE + 1);
  gridSize = min(gridSize, GRID_SIZE);
  d_ActivationFunction<<< gridSize, BLOCK_SIZE, 0, cudaStream >>>(d_A, d_row, d_col);
}

__global__ void d_AddToDiagonal(float *d_A, int *d_row, int *d_col, float *d_alfa){
  __shared__ int m, n, area;
  m = (*d_row);
  n = (*d_col);
  area = gridDim.x*blockDim.x;
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  while (id < m && id < n){
    int cell = id + id*n;
    d_A[cell] += (*d_alfa);
    id += area;
    cell = id+id*n;
  }
}

void addToDiagonal(cudaStream_t cudaStream, float *d_A, int row, int col, int *d_row, int *d_col, float *d_alfa){
  int gridSize = (col/BLOCK_SIZE + 1);
  gridSize = min(gridSize, GRID_SIZE);
  // cout<<gridSize<<" "<<BLOCK_SIZE<<"\n";
  d_AddToDiagonal<<< gridSize, BLOCK_SIZE, 0, cudaStream >>>(d_A, d_row, d_col, d_alfa);
}
//
// ConfigTest readConfigTest(string configFileName){
//   ifstream configFile(configFileName);
//
//   ConfigTest config;
//   string dataName;
//   configFile >> dataName;
//   config.xFileName = "data/"+dataName+"/test/file_x.bin";
//   config.yFileName = "data/"+dataName+"/test/file_y.bin";
//   configFile >> config.row >> config.col >> config.classNum >> config.hiddenNeuron;
//   configFile >> config.alpha;
//   config.wInputFileName = "weight/"+dataName+"/w-in-"+
//   to_string(config.hiddenNeuron)+".bin";
//   config.wOutputFileName = "weight/"+dataName+"/w-out-"+
//   to_string(config.hiddenNeuron)+".bin";
//   configFile >> config.accuracyFileName;
//   config.accuracyFileName = "report/"+dataName+"/"+config.accuracyFileName;
//
//   cout << "Using " << dataName <<"\n";
//   cout << "Total Rows : " << config.row <<"; Cols : "<< config.col<<"; ";
//   cout << "Class : " << config.classNum<<"\n";
//   cout << "Hidden Neuron : " << config.hiddenNeuron <<"\n";
//   cout << "W-input Data: " << config.wInputFileName <<"\n";
//   cout << "W-output Data: " << config.wOutputFileName <<"\n";
//   cout << "Test X Data: " << config.xFileName <<"\n";
//   cout << "Test Y Data: " << config.yFileName <<"\n";
//   return config;
// }
//
// void writeMatrixfToFileBinary(std::string fileName, float* matrix, int m, int n){
//   fileName = fileName + ".bin";
//   printf("Writing Matrix to file %s\n", fileName);
//   printf("Rows = %d, Col = %d\n", m, n);
//
//   ofstream outFile(fileName, ios::trunc | ios::binary);
//   outFile.write((char *) matrix, m*n*sizeof(float));
//   outFile.close();
// }
//
// void writeRunningTimeData(string fileName, RunningTimeData rt){
//   printf("Writing output data to file %s\n", fileName);
//   FILE *outFile;
//   if( access( fileName.c_str(), F_OK ) != -1 ) {
//     outFile = fopen(fileName.c_str(), "a");
//   } else {
//     outFile = fopen(fileName.c_str(), "wb");
//     fprintf(outFile, "NP,ROW,COL,HIDDEN_NEURON,READ_TIME,WRITE_TIME,GEN_W_TIME,MAX_H,MAX_A,MAX_W,COMBINE_W,TOTAL,REAL_TOTAL\n");
//   }
//   fprintf(outFile, "%d,%d,%d,%d,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf\n", rt.np,
//   rt.row, rt.col, rt.hiddenNeuron, rt.readDataTime,
//   rt.writeDataTime, rt.generateWeightTime, rt.maxH,
//   rt.maxA, rt.maxW, rt.combineW,  rt.totalTime,
//   rt.realTotalTime);
//
//   fclose(outFile);
// }
//
// void writeAccuracyData(string fileName, AccuracyData accuracyData){
//   printf("Writing accuracy data to file %s\n", fileName);
//   FILE *outFile;
//   if( access( fileName.c_str(), F_OK ) != -1 ) {
//     outFile = fopen(fileName.c_str(), "a");
//   } else {
//     outFile = fopen(fileName.c_str(), "wb");
//     fprintf(outFile, "RMSE,TRUE_ACCURACY\n");
//   }
//   fprintf(outFile, "%.5lf,%.5lf\n", accuracyData.RMSE, accuracyData.TrueAccuracy);
//   fclose(outFile);
// }
