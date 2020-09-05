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

  /*
    allocate global memories
    variables if any
    1. input weight
    2. output weight
  */
  conf.alpha = 1/conf.alpha;
  float *d_alfa;
  cudaMalloc(&d_alfa, sizeof(float));
  cudaMemcpyAsync(d_alfa, &conf.alpha, sizeof(float), cudaMemcpyHostToDevice);

  int col1 = conf.col + 1;
  int *d_row, *d_col, *d_col1, *d_hiddenNeuron;
  bool *d_true, *d_false;
  bool ttrue = true, ffalse = false;
  magma_malloc((void **)&d_row, sizeof(int));
  magma_malloc((void **)&d_col, sizeof(int));
  magma_malloc((void **)&d_col1, sizeof(int));
  magma_malloc((void **)&d_hiddenNeuron, sizeof(int));
  magma_malloc((void **)&d_true, sizeof(bool));
  magma_malloc((void **)&d_false, sizeof(bool));
  cudaMemcpy(d_col, &conf.col, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col1, &col1, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_hiddenNeuron, &conf.hiddenNeuron, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_true, &ttrue, sizeof(bool), cudaMemcpyHostToDevice);
  cudaMemcpy(d_false, &ffalse, sizeof(bool), cudaMemcpyHostToDevice);

  float *d_W;
  err = magma_smalloc(&d_W, (col1)*conf.hiddenNeuron);
  if (rank == ROOT){
    float *W, *Wt;
    err = magma_smalloc_cpu(&W, (col1)*conf.hiddenNeuron);
    err = magma_smalloc_cpu(&Wt, (col1)*conf.hiddenNeuron);
    magma_int_t ione=1;
    magma_int_t ISEED[4]={0, 1, 2, 3};
    magma_int_t wSize = (col1)*conf.hiddenNeuron;
    lapackf77_slarnv(&ione, ISEED, &wSize, Wt);
    transpose_smatrix(Wt, W, col1, conf.hiddenNeuron);
    magma_ssetmatrix(col1, conf.hiddenNeuron, W, col1, d_W, col1, queue);
    magma_free_cpu(Wt);
  }
  MPI_Bcast(d_W, (col1)*conf.hiddenNeuron, MPI_FLOAT, ROOT, MPI_COMM_WORLD);


  /*
    row size and row offset per subsection (per thread per process)
    row offset is needed for mpi-read
  */
  int row, rowOffset;
  getRowSplitSize(conf.row, numProcs, rank, &row, &rowOffset);
  std::printf("subIdx %d: row %d and offset %d\n", rank, row, rowOffset);
  cudaMemcpy(d_row, &row, sizeof(int), cudaMemcpyHostToDevice);

  /*
    Allocate memory
  */
  float *X, *Y;
  float *d_X, *d_Y, *d_H, *d_A;

  err =  magma_smalloc_cpu(&X, row*(col1));
  err =  magma_smalloc_cpu(&Y, row*(conf.classNum));
  err =  magma_smalloc(&d_X, row*(col1));
  err =  magma_smalloc(&d_Y, row*(conf.classNum));
  err =  magma_smalloc(&d_H, row*(conf.hiddenNeuron));
  err =  magma_smalloc(&d_A, row*(row));

  // Read traing X and Y sub-matrices,
  read_smatrix(MPI_COMM_WORLD, conf.xFileName, X, row, rowOffset, col1);
  magma_ssetmatrix(row, col1, X, row, d_X, row, queue);
  read_smatrix(MPI_COMM_WORLD, conf.yFileName, Y, row, rowOffset, conf.classNum);
  magma_ssetmatrix(row, conf.classNum, Y, row, d_Y, row, queue);
  /*
    1. H(conf.row,conf.hiddenNeuron) = X(conf.row,conf.col+1)*W(conf.col+1, conf.hiddenNeuron)
    2. activation Function (H)
  */
  double gpuTime = magma_sync_wtime ( queue );
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, row, conf.hiddenNeuron, col1,
    1.0 , d_X, row, d_W, col1, 0., d_H, row, queue);
  activationFunction(cudaStream, d_H, row, conf.hiddenNeuron, d_row, d_hiddenNeuron);
  rt.maxW = magma_sync_wtime ( queue ) - gpuTime;
  std::printf("Rank %d: calculate W %.9lf seconds\n", rank, rt.maxW);

  /*
    3. A(conf.row, conf.row) = H*Ht
    4. Adiag += 1/alfa
  */
  gpuTime = magma_sync_wtime ( queue );
  magma_sgemm(MagmaNoTrans, MagmaNoTrans, row, row, conf.hiddenNeuron,
    1.0 , d_H, row, d_H, row, 0., d_A, row, queue);
  addToDiagonal(cudaStream, d_A, row, row, d_row, d_row, d_alfa, d_true);
  rt.maxA = magma_sync_wtime ( queue ) - gpuTime;
  std::printf("Rank %d: calculate A %.9lf seconds\n", rank, rt.maxA);

  


  // magma_free(d_row);
  // magma_free(d_col);
  // magma_free(d_col1);
  // magma_free(d_hiddenNeuron);
  // magma_free(d_X);
  // magma_free(d_Y);
  // magma_free(d_H);
  // magma_free(d_A);
  // magma_free(d_alfa);
  magma_queue_destroy(queue);
  magma_finalize();
  MPI_Finalize();
}
