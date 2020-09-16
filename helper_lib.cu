#include <algorithm>
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
#include "magma_v2.h"
#include "magma_lapack.h"

#define BLOCK_SIZE 256
#define GRID_SIZE 1000
#define FLT_EPSILON 1e-5


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
  config.alpha = stof(argv[6]);
  config.wInputFileName = "weight/"+dataName+"/w-in-"+
    to_string(config.hiddenNeuron)+".bin";
  config.wOutputFileName = "weight/"+dataName+"/w-out-"+
  to_string(config.hiddenNeuron)+".bin";
  config.subCount = atoi(argv[7]);
  config.runningTimeFileName = argv[8];
  config.runningTimeFileName = "report/"+dataName+"/"+config.runningTimeFileName;

  cout << "Using " << dataName <<"\n";
  cout << "Total Rows : " << config.row <<"; Cols : "<< config.col<<"; ";
  cout << "Class : " << config.classNum<<"\n";
  cout << "Hidden Neuron : " << config.hiddenNeuron <<"\n";
  return config;
}

ConfigTest readConfigTest(char **argv){
  ConfigTest config;
  string dataName = argv[1];
  config.xFileName = "data/"+dataName+"/test/file_x.bin";
  config.yFileName = "data/"+dataName+"/test/file_y.bin";
  config.row = atoi(argv[2]);
  config.col = atoi(argv[3]);
  config.classNum = atoi(argv[4]);
  config.hiddenNeuron = atoi(argv[5]);
  config.alpha = stof(argv[6]);
  config.wInputFileName = "weight/"+dataName+"/w-in-"+
  to_string(config.hiddenNeuron)+".bin";
  config.wOutputFileName = "weight/"+dataName+"/w-out-"+
  to_string(config.hiddenNeuron)+".bin";
  config.subCount = atoi(argv[7]);
  config.accuracyFileName = argv[8];
  config.accuracyFileName = "report/"+dataName+"/"+config.accuracyFileName;

  cout << "Using " << dataName <<"\n";
  cout << "Total Rows : " << config.row <<"; Cols : "<< config.col<<"; ";
  cout << "Class : " << config.classNum<<"\n";
  cout << "Hidden Neuron : " << config.hiddenNeuron <<"\n";
  cout << "W-input Data: " << config.wInputFileName <<"\n";
  cout << "W-output Data: " << config.wOutputFileName <<"\n";
  cout << "Test X Data: " << config.xFileName <<"\n";
  cout << "Test Y Data: " << config.yFileName <<"\n";
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

void read_smatrix(MPI_Comm comm, std::string fileName, float *arr, int row,
  int rowOffset, int col, bool transpose){
  MPI_Status status;
  MPI_File file;
  char *fileNameChar = const_cast<char*>(fileName.c_str());
  MPI_File_open(comm, fileNameChar, MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
  if (transpose){
    float *arrt;
    arrt = (float*)malloc(row*col*sizeof(float));
    MPI_File_read_at(file, rowOffset*col*sizeof(float), arrt, row*col, MPI_FLOAT, &status);
    transpose_smatrix(arrt, arr, row, col);
    free(arrt);
  } else {
    MPI_File_read_at(file, rowOffset*col*sizeof(float), arr, row*col, MPI_FLOAT, &status);
  }

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
    float tmp = exp(-d_A[i]);
    d_A[i] = 1.0 / (1.0 + tmp);
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

void getPseudoInverse(magma_queue_t queue, float *A, float *d_Ainv, int row,
  int col){
    magma_int_t nb = magma_get_sgesvd_nb(row, col);
    magma_int_t mn = min(row, col), mx = max(row, col);
    magma_int_t lwork = -1;
    float *u;
    float *d_u;
    float *vt;
    float *d_vt;
    float *sig;
    float *sigmat;
    float *d_sigmat;
    float *d_temp;
    float *work;
    int *iwork;
    magma_int_t err ;
    magma_int_t info;
    err = magma_smalloc_cpu(&u, row*row);
    err = magma_smalloc_cpu(&sig, mn);
    err = magma_smalloc_cpu(&sigmat, row*col);
    err = magma_smalloc_cpu(&vt, col*col);
    err = magma_smalloc_cpu(&work, 1);
    err = magma_imalloc_cpu(&iwork, 8*mn);
    err = magma_smalloc(&d_sigmat, col*row);
    err = magma_smalloc(&d_vt, col*col);
    err = magma_smalloc(&d_u, row*row);
    magma_smalloc(&d_temp, col*row);

    magma_sgesdd(MagmaAllVec, row, col, A, row, sig, u, row, vt, col,
    work, lwork, iwork, &info);
    lwork = work[0]+1;
    err = magma_smalloc_cpu(&work, lwork);

    // magma_sprint(row, col, A, row);
    magma_sgesdd(MagmaAllVec, row, col, A, row, sig, u, row, vt, col,
    work, lwork, iwork, &info);

    #pragma omp parallel for collapse(2)
    for(int i=0;i<col;i++){
      for(int j=0;j<row;j++){
        sigmat[i*row+j]=0;
      }
    }

    #pragma omp parallel for
    for(int i=0;i<mn;i++){
      if (sig[i]>FLT_EPSILON || sig[i]<-FLT_EPSILON){
        sigmat[i*col+i]=1/sig[i];
      }
    }
    magma_ssetmatrix(col, row, sigmat, col, d_sigmat, col, queue);
    magma_ssetmatrix(col, col, vt, col, d_vt, col, queue);
    magma_ssetmatrix(row, row, u, row, d_u, row, queue);

    magma_sgemm(MagmaTrans, MagmaNoTrans, col, row, col, 1, d_vt, col,
    d_sigmat, col, 0, d_temp, col, queue);
    magma_sgemm(MagmaNoTrans, MagmaTrans, col, row, row, 1, d_temp, col,
    d_u, row, 0, d_Ainv, col, queue);
    magma_sync_wtime(queue);

    magma_free(d_u);
    magma_free(d_vt);
    magma_free(d_temp);
    magma_free(d_sigmat);
    magma_free_cpu(u);
    magma_free_cpu(vt);
    magma_free_cpu(sig);
    magma_free_cpu(work);
    magma_free_cpu(iwork);
}

void write_smatrix(std::string fileName, float* matrix, int m, int n){
  printf("Writing Matrix to file %s\n", fileName);
  printf("Rows = %d, Col = %d\n", m, n);

  ofstream outFile(fileName, ios::trunc | ios::binary);
  outFile.write((char *) matrix, m*n*sizeof(float));
  outFile.close();
}

void writeRunningTimeData(string fileName, RunningTimeData rt){
  printf("Writing output data to file %s\n", fileName);
  FILE *outFile;
  if( access( fileName.c_str(), F_OK ) != -1 ) {
    outFile = fopen(fileName.c_str(), "a");
  } else {
    outFile = fopen(fileName.c_str(), "wb");
    fprintf(outFile, "NP,ROW,COL,HIDDEN_NEURON,READ_TIME,GEN_W_TIME,MAX_H,MAX_A,MAX_W,COMBINE_W,TOTAL\n");
  }
  fprintf(outFile, "%d,%d,%d,%d,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf,%.5lf\n", rt.np,
  rt.row, rt.col, rt.hiddenNeuron, rt.readDataTime, rt.maxH,
  rt.maxA, rt.maxW, rt.combineW,  rt.totalTime);
  fclose(outFile);
}

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
