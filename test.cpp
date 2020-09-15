#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#include "helper_lib.h"
#include "magma_v2.h"
#include "magma_lapack.h"

int main(int argc, char **argv){
  MPI_Status status;
  int rank;
  int numProcs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ConfigTest conf = readConfigTest(argv);

  if(numProcs > 1){
    return 0;
  }

  magma_init();
  magma_queue_t queue = NULL;
  magma_queue_create(0, &queue);
  magma_int_t err;
  cudaStream_t cudaStream = magma_queue_get_cuda_stream(queue);

  // memory allocation
  float *X, *Y, *Winp, *Wout, *Ypred, *Ypredt;
  float *d_X, *d_Winp, *d_Wout, *d_H, *d_Ypred;
  int *Yvec, *YpredVec;
  magma_smalloc_cpu(&X, conf.row*(conf.col+1));
  magma_smalloc_cpu(&Y, conf.row*conf.classNum);
  magma_smalloc_cpu(&Ypred, conf.row*conf.classNum);
  magma_smalloc_cpu(&Ypredt, conf.row*conf.classNum);
  magma_smalloc_cpu(&Winp, (conf.col+1)*conf.hiddenNeuron);
  magma_smalloc_cpu(&Wout, conf.hiddenNeuron*conf.classNum);
  magma_imalloc_cpu(&Yvec, conf.row);
  magma_imalloc_cpu(&YpredVec, conf.row);
  magma_smalloc(&d_X, conf.row*(conf.col+1));
  magma_smalloc(&d_Ypred, conf.row*conf.classNum);
  magma_smalloc(&d_Winp, (conf.col+1)*conf.hiddenNeuron);
  magma_smalloc(&d_Wout, conf.hiddenNeuron*conf.classNum);
  magma_smalloc(&d_H, conf.row*conf.hiddenNeuron);

  int *row, *hiddenNeuron;
  cudaMallocManaged(&row, sizeof(int));
  cudaMallocManaged(&hiddenNeuron, sizeof(int));
  *row = conf.row;
  *hiddenNeuron = conf.hiddenNeuron;

  // read data and the generated weights
  read_smatrix(MPI_COMM_WORLD, conf.xFileName, X, conf.row, 0, conf.col+1, true);
  read_smatrix(MPI_COMM_WORLD, conf.yFileName, Y, conf.row, 0, conf.classNum, false);
  read_smatrix(MPI_COMM_WORLD, conf.wInputFileName, Winp, conf.col+1, 0, conf.hiddenNeuron, false);
  read_smatrix(MPI_COMM_WORLD, conf.wOutputFileName, Wout, conf.hiddenNeuron, 0, conf.classNum, false);

  /*
    get the class of Y
  */
  for(int i=0;i<conf.row;i++){
    for(int j=0;j<conf.classNum;j++){
      if (Y[i*conf.classNum + j] > 0.5){
        Yvec[i]=j;
        break;
      }
    }
  }

  magma_ssetmatrix(conf.row, conf.col+1, X, conf.row, d_X, conf.row, queue);
  magma_ssetmatrix(conf.col+1, conf.hiddenNeuron, Winp, conf.col+1, d_Winp, conf.col+1, queue);
  magma_ssetmatrix(conf.hiddenNeuron, conf.classNum, Wout, conf.hiddenNeuron, d_Wout, conf.hiddenNeuron, queue);

  // FeedForward
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, conf.row, conf.hiddenNeuron, conf.col+1,
    1.0 , d_X, conf.row, d_Winp, conf.col+1, 0., d_H, conf.row, queue);
  magma_sync_wtime(queue);
  activationFunction(cudaStream, d_H, conf.row, conf.hiddenNeuron, row, hiddenNeuron);
  cudaDeviceSynchronize();
  // magma_sprint_gpu(conf.row, conf.hiddenNeuron, d_H, conf.row, queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, conf.row, conf.classNum, conf.hiddenNeuron,
    1.0 , d_H, conf.row, d_Wout, conf.hiddenNeuron, 0., d_Ypred, conf.row, queue);
  magma_sgetmatrix(conf.row, conf.classNum, d_Ypred, conf.row, Ypredt, conf.row, queue);
  magma_sync_wtime(queue);
  transpose_smatrix(Ypredt, Ypred, conf.classNum, conf.row);

  /*
    determining predicted class
  */
  for(int i=0;i<conf.row;i++){
    float maxVal = -9999;
    int cls = 0;
    for(int j=0;j<conf.classNum;j++){
      if (maxVal < Ypred[i*conf.classNum + j]){
        maxVal = Ypred[i*conf.classNum + j];
        cls = j;
      }
    }
    YpredVec[i]=cls;
  }

  /*
    compute accuracy
  */
  double trueCount = 0;
  double total = conf.row;
  for(int i=0;i<conf.row;i++){
    if (Yvec[i]==YpredVec[i]){
      trueCount +=1;
    }
  }
  printf("Accuracy = %.6lf\n", trueCount/total);

  magma_queue_destroy(queue);
  magma_finalize();
  MPI_Finalize();
}
