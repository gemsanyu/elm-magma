#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#include "helper_lib.h"
#include "magma_v2.h"
#include "magma_lapack.h"

/*
  numProcs = 1 always
  test data might be divided into subs in case the test data is too huge
*/

int main(int argc, char **argv){
  MPI_Status status;
  int rank;
  int numProcs;
  MPI_Init(&argc, &argv);
  ConfigTest conf = readConfigTest(argv);

  magma_init();
  magma_queue_t queue = NULL;
  magma_queue_create(0, &queue);
  magma_int_t err;
  cudaStream_t cudaStream = magma_queue_get_cuda_stream(queue);

  int rowEst = conf.row/conf.subCount + 1;

  // memory allocation
  float *X, *Y, *Winp, *Wout, *Ypred, *Ypredt;
  float *d_X, *d_Winp, *d_Wout, *d_H, *d_Ypred;
  int *Yvec, *YpredVec;
  magma_smalloc_cpu(&X, rowEst*(conf.col+1));
  magma_smalloc_cpu(&Y, rowEst*conf.classNum);
  magma_smalloc_cpu(&Ypred, rowEst*conf.classNum);
  magma_smalloc_cpu(&Ypredt, rowEst*conf.classNum);
  magma_smalloc_cpu(&Winp, (conf.col+1)*conf.hiddenNeuron);
  magma_smalloc_cpu(&Wout, conf.hiddenNeuron*conf.classNum);
  magma_imalloc_cpu(&Yvec, rowEst);
  magma_imalloc_cpu(&YpredVec, rowEst);
  magma_smalloc(&d_X, rowEst*(conf.col+1));
  magma_smalloc(&d_Ypred, rowEst*conf.classNum);
  magma_smalloc(&d_Winp, (conf.col+1)*conf.hiddenNeuron);
  magma_smalloc(&d_Wout, conf.hiddenNeuron*conf.classNum);
  magma_smalloc(&d_H, rowEst*conf.hiddenNeuron);

  int *row, *hiddenNeuron, *rowOffset;
  cudaMallocManaged(&row, sizeof(int));
  cudaMallocManaged(&rowOffset, sizeof(int));
  cudaMallocManaged(&hiddenNeuron, sizeof(int));
  *hiddenNeuron = conf.hiddenNeuron;

  // read the generated weights
  read_smatrix(MPI_COMM_WORLD, conf.wInputFileName, Winp, conf.col+1, 0, conf.hiddenNeuron, false);
  read_smatrix(MPI_COMM_WORLD, conf.wOutputFileName, Wout, conf.hiddenNeuron, 0, conf.classNum, false);
  magma_ssetmatrix(conf.col+1, conf.hiddenNeuron, Winp, conf.col+1, d_Winp, conf.col+1, queue);
  magma_ssetmatrix(conf.hiddenNeuron, conf.classNum, Wout, conf.hiddenNeuron, d_Wout, conf.hiddenNeuron, queue);


  double trueCount = 0;
  double total = conf.row;
  for(int i=0; i<conf.subCount;i++){

    // read the test data
    getRowSplitSize(conf.row, conf.subCount, i, row, rowOffset);
    read_smatrix(MPI_COMM_WORLD, conf.xFileName, X, (*row), (*rowOffset), conf.col+1, true);
    read_smatrix(MPI_COMM_WORLD, conf.yFileName, Y, (*row), (*rowOffset), conf.classNum, false);

    /*
      get the class of Y
    */
    for(int i=0;i<(*row);i++){
      for(int j=0;j<conf.classNum;j++){
        if (Y[i*conf.classNum + j] > 0.5){
          Yvec[i]=j;
          break;
        }
      }
    }

    magma_ssetmatrix((*row), conf.col+1, X, (*row), d_X, (*row), queue);

    // FeedForward
    magma_sgemm(MagmaNoTrans, MagmaNoTrans, (*row), conf.hiddenNeuron, conf.col+1,
      1.0 , d_X, (*row), d_Winp, conf.col+1, 0., d_H, (*row), queue);
    magma_sync_wtime(queue);
    activationFunction(cudaStream, d_H, (*row), conf.hiddenNeuron, row, hiddenNeuron);
    cudaDeviceSynchronize();

    // printf("%d %d %d\n", conf.row, *row, rowEst);
    magma_sgemm(MagmaNoTrans, MagmaNoTrans, (*row), conf.classNum, conf.hiddenNeuron,
      1.0 , d_H, (*row), d_Wout, conf.hiddenNeuron, 0., d_Ypred, (*row), queue);
    magma_sync_wtime(queue);
    magma_sgetmatrix((*row), conf.classNum, d_Ypred, (*row), Ypredt, (*row), queue);
    magma_sync_wtime(queue);
    transpose_smatrix(Ypredt, Ypred, conf.classNum, (*row));

    /*
      determining predicted class
    */
    for(int i=0;i<(*row);i++){
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
    for(int i=0;i<(*row);i++){
      if (Yvec[i]==YpredVec[i]){
        trueCount +=1;
      }
    }
  }
  printf("Accuracy = %.6lf\n", trueCount/total);

  magma_queue_destroy(queue);
  magma_finalize();
  MPI_Finalize();
}
