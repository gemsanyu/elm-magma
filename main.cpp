#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

#include "helper_lib.h"
#include "magma_v2.h"
#include "magma_lapack.h"

#define ROOT 0

int main(int argc, char **argv){
  MPI_Status status;
  int rank;
  int numProcs;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Config conf = readConfig(argv);

  magma_init();
  magma_queue_t queue = NULL;
  magma_queue_create(0, &queue);
  magma_int_t err;
  cudaStream_t cudaStream = magma_queue_get_cuda_stream(queue);

  RunningTimeData rt;
  double gpuTime;
  /*
    allocate global memories
    variables if any
    1. input weight
    2. output weight
  */
  float *alfa;
  int *col1, *row, *col, *hiddenNeuron;
  cudaMallocManaged(&alfa, sizeof(float));
  cudaMallocManaged(&col1, sizeof(int));
  cudaMallocManaged(&col, sizeof(int));
  cudaMallocManaged(&row, sizeof(int));
  cudaMallocManaged(&hiddenNeuron, sizeof(int));
  *alfa = 1/conf.alpha;
  *col1 = conf.col + 1;
  *col = conf.col;
  *hiddenNeuron = conf.hiddenNeuron;


  float *W;
  cudaMallocManaged(&W, (*col1)*(conf.hiddenNeuron)*sizeof(float));
  if (rank == ROOT){
    magma_int_t ione=1;
    magma_int_t ISEED[4]={0, 1, 2, 3};
    magma_int_t wSize = (*col1)*conf.hiddenNeuron;
    lapackf77_slarnv(&ione, ISEED, &wSize, W);
  }
  MPI_Bcast(W, (*col1)*conf.hiddenNeuron, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

  /*
    row size and row offset per subsection (per thread per process)
    row offset is needed for mpi-read
  */
  int rowOffset;
  getRowSplitSize(conf.row, numProcs, rank, row, &rowOffset);
  std::printf("subIdx %d: row %d and offset %d\n", rank, *row, rowOffset);

  /*
    Allocate memory
  */
  float *X, *Y, *H, *A;
  cudaMallocManaged(&X, (*row)*(*col1)*sizeof(float));
  cudaMallocManaged(&Y, (*row)*(conf.classNum)*sizeof(float));
  cudaMallocManaged(&H, (*row)*(conf.hiddenNeuron)*sizeof(float));
  cudaMallocManaged(&A, (*row)*(*row)*sizeof(float));

  // Read traing X and Y sub-matrices,
  read_smatrix(MPI_COMM_WORLD, conf.xFileName, X, *row, rowOffset, *col1);
  read_smatrix(MPI_COMM_WORLD, conf.yFileName, Y, *row, rowOffset, conf.classNum);

  /*
    1. H(conf.row,conf.hiddenNeuron) = X(conf.row,conf.col+1)*W(conf.col+1, conf.hiddenNeuron)
    2. activation Function (H)
  */
  gpuTime = magma_sync_wtime(queue);
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, *row, conf.hiddenNeuron, *col1,
    1.0 , X, *row, W, *col1, 0., H, *row, queue);
  magma_sync_wtime(queue);
  activationFunction(cudaStream, H, (*row), conf.hiddenNeuron, row, hiddenNeuron);
  cudaStreamSynchronize(cudaStream);
  rt.maxW = magma_sync_wtime(queue) - gpuTime;
  // magma_sprint_gpu(*row, conf.hiddenNeuron, H, *row, queue);
  std::printf("Rank %d: calculate W %.9lf seconds\n", rank, rt.maxW);

  /*
    3. A(conf.row, conf.row) = H*Ht
    4. Adiag += 1/alfa
  */
  gpuTime = magma_sync_wtime ( queue );
  magma_sgemm(MagmaNoTrans, MagmaTrans, *row, *row, conf.hiddenNeuron,
    1.0 , H, *row, H, *row, 0., A, *row, queue);
  magma_sync_wtime(queue);
  addToDiagonal(cudaStream, A, *row, *row, row, row, alfa, true);
  cudaStreamSynchronize(cudaStream);
  rt.maxA = magma_sync_wtime (queue) - gpuTime;
  std::printf("Rank %d: calculate A %.9lf seconds\n", rank, rt.maxA);




  magma_free(d_row);
  magma_free(d_col);
  magma_free(d_col1);
  magma_free(d_hiddenNeuron);
  magma_free(d_X);
  magma_free(d_Y);
  magma_free(d_H);
  magma_free(d_A);
  magma_free(d_alfa);
  magma_queue_destroy(queue);
  magma_finalize();
  MPI_Finalize();
}
