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
  rt.np = 0;
  rt.row = 0;
  rt.col = 0;
  rt.hiddenNeuron = 0;
  rt.readDataTime = 0;
  rt.maxA = 0;
  rt.maxH = 0;
  rt.maxW = 0;
  rt.memoryAllocation = 0;
  rt.totalTime = 0;
  double gpuTime;
  /*
    allocate global memories
    variables if any
    1. input weight
    2. output weight
  */
  float *alfa;
  int *col1, *row, *col, *hiddenNeuron;
  gpuTime = magma_sync_wtime(queue);
  cudaMallocManaged(&alfa, sizeof(float));
  cudaMallocManaged(&col1, sizeof(int));
  cudaMallocManaged(&col, sizeof(int));
  cudaMallocManaged(&row, sizeof(int));
  cudaMallocManaged(&hiddenNeuron, sizeof(int));
  rt.memoryAllocation = magma_sync_wtime(queue)-gpuTime;
  *alfa = 1/conf.alpha;
  *col1 = conf.col + 1;
  *col = conf.col;
  *hiddenNeuron = conf.hiddenNeuron;

  /*
    calculating row estimate per subdata
    to reduce the number of times of allocating and freeing memory
  */
  int totalSubCount = numProcs*conf.subCount;
  int rowEst = conf.row/totalSubCount + 1;

  /*
    Allocate memory
    d_Aproc, d_AWproc is to combine all subcomputation per process
    d_Acombined is to combine all subcomputation from each process
  */
  float *X, *Y, *A, *Acombined, *Winp, *Wout;
  float *d_X, *d_Y, *d_H, *d_A, *d_Ainv, *d_HtY, *d_W, *d_Winp, *d_Wout;
  float *d_Aproc, *d_AWproc;
  float *d_Acombined, *d_AcombinedInv, *d_AW, *d_AWcombined;
  gpuTime = magma_sync_wtime(queue);
  magma_smalloc_cpu(&X, rowEst*(*col1));
  magma_smalloc_cpu(&Y, rowEst*conf.classNum);
  magma_smalloc_cpu(&A, conf.hiddenNeuron*conf.hiddenNeuron);
  magma_smalloc_cpu(&Winp, (*col1)*conf.hiddenNeuron);
  magma_smalloc(&d_X, rowEst*(*col1));
  magma_smalloc(&d_Y, rowEst*conf.classNum);
  magma_smalloc(&d_H, rowEst*conf.hiddenNeuron);
  magma_smalloc(&d_A, conf.hiddenNeuron*conf.hiddenNeuron);
  magma_smalloc(&d_Aproc, conf.hiddenNeuron*conf.hiddenNeuron);
  magma_smalloc(&d_Ainv, conf.hiddenNeuron*conf.hiddenNeuron);
  magma_smalloc(&d_HtY, conf.hiddenNeuron*conf.classNum);
  magma_smalloc(&d_W, conf.hiddenNeuron*conf.classNum);
  magma_smalloc(&d_AW, conf.hiddenNeuron*conf.classNum);
  magma_smalloc(&d_AWproc, conf.hiddenNeuron*conf.classNum);
  magma_smalloc(&d_Winp, (*col1)*conf.hiddenNeuron);

  cudaMemset(d_Aproc, 0, conf.hiddenNeuron*conf.hiddenNeuron*sizeof(float));
  cudaMemset(d_AWproc, 0, conf.hiddenNeuron*conf.classNum*sizeof(float));
  /*
    ROOT only variables
    for combining purposes
  */
  if (rank == ROOT){
    magma_smalloc(&d_Acombined, conf.hiddenNeuron*conf.hiddenNeuron);
    magma_smalloc(&d_AcombinedInv, conf.hiddenNeuron*conf.hiddenNeuron);
    magma_smalloc(&d_Wout, conf.hiddenNeuron*conf.classNum);
    magma_smalloc(&d_AWcombined, conf.hiddenNeuron*conf.classNum);
    magma_smalloc_cpu(&Acombined, conf.hiddenNeuron*conf.hiddenNeuron);
    magma_smalloc_cpu(&Wout, conf.hiddenNeuron*conf.classNum);
    cudaMemset(d_Acombined, 0, conf.hiddenNeuron*conf.hiddenNeuron*sizeof(float));
    cudaMemset(d_Wout, 0, conf.hiddenNeuron*conf.classNum*sizeof(float));
    cudaMemset(d_AWcombined, 0, conf.hiddenNeuron*conf.classNum*sizeof(float));
  }

  rt.memoryAllocation += (magma_sync_wtime(queue) - gpuTime);

  // Read weight input
  gpuTime = magma_sync_wtime(queue);
  read_smatrix(MPI_COMM_WORLD, conf.wInputFileName, Winp, *col1, 0, conf.hiddenNeuron, false);
  magma_ssetmatrix(*col1, conf.hiddenNeuron, Winp, *col1, d_Winp, *col1, queue);
  rt.readDataTime = magma_sync_wtime(queue) - gpuTime;

  for(int i=0;i<conf.subCount;i++){
    /*
      row size and row offset per subsection (per thread per process)
      row offset is needed for mpi-read
    */
    int subIdx = rank*conf.subCount + i;
    int rowOffset;
    getRowSplitSize(conf.row, totalSubCount, subIdx, row, &rowOffset);
    // std::printf("Rank %d: subIdx %d: row %d and offset %d\n", rank, subIdx, *row, rowOffset);

    // Read traing X and Y sub-matrices,
    gpuTime = magma_sync_wtime(queue);
    read_smatrix(MPI_COMM_WORLD, conf.xFileName, X, *row, rowOffset, *col1, true);
    read_smatrix(MPI_COMM_WORLD, conf.yFileName, Y, *row, rowOffset, conf.classNum, true);
    magma_ssetmatrix(*row, *col1, X, *row, d_X, *row, queue);
    magma_ssetmatrix(*row, conf.classNum, Y, *row, d_Y, *row, queue);
    rt.readDataTime += (magma_sync_wtime(queue) - gpuTime);

    /*
      1. H(conf.row,conf.hiddenNeuron) = X(conf.row,conf.col+1)*W(conf.col+1, conf.hiddenNeuron)
      2. activation Function (H)
    */
    gpuTime = magma_sync_wtime(queue);
    magma_sgemm(MagmaNoTrans, MagmaNoTrans, *row, conf.hiddenNeuron, *col1,
      1.0 , d_X, *row, d_Winp, *col1, 0., d_H, *row, queue);
    magma_sync_wtime(queue);
    // magma_sprint_gpu(*row, conf.hiddenNeuron, d_H, *row, queue);
    activationFunction(cudaStream, d_H, (*row), conf.hiddenNeuron, row, hiddenNeuron);
    cudaStreamSynchronize(cudaStream);
    rt.maxH += (magma_sync_wtime(queue) - gpuTime);
    // magma_sprint_gpu(*row, conf.hiddenNeuron, d_H, *row, queue);


    /*
      3. A(hn, hn) = Ht*H
      4. Adiag += 1/alfa
    */
    gpuTime = magma_sync_wtime ( queue );
    magma_sgemm(MagmaTrans, MagmaNoTrans, conf.hiddenNeuron, conf.hiddenNeuron, (*row),
      1.0 , d_H, *row, d_H, *row, 0., d_A, conf.hiddenNeuron, queue);
    magma_sync_wtime(queue);
    addToDiagonal(cudaStream, d_A, *hiddenNeuron, *hiddenNeuron, hiddenNeuron, hiddenNeuron, alfa);
    cudaStreamSynchronize(cudaStream);
    rt.maxA += (magma_sync_wtime (queue) - gpuTime);

    /*
      5. copy A to Host first, A inverse
      6. HtY = Ht*Y
      7. Wout = Ainv * HtY
    */
    gpuTime = magma_sync_wtime(queue);
    magma_sgetmatrix(*hiddenNeuron, *hiddenNeuron, d_A, *hiddenNeuron, A, *hiddenNeuron, queue);
    magma_sync_wtime(queue);
    getPseudoInverse(queue, A, d_Ainv, *hiddenNeuron, *hiddenNeuron);
    magma_sgemm(MagmaTrans, MagmaNoTrans, conf.hiddenNeuron, conf.classNum, (*row),
      1.0 , d_H, *row, d_Y, *row, 0., d_HtY, conf.hiddenNeuron, queue);
    magma_sgemm(MagmaNoTrans, MagmaNoTrans, conf.hiddenNeuron, conf.classNum, conf.hiddenNeuron,
      1.0 , d_Ainv, conf.hiddenNeuron, d_HtY, conf.hiddenNeuron, 0., d_W, conf.hiddenNeuron, queue);
    rt.maxW += (magma_sync_wtime(queue) - gpuTime);

    /*
      8. Combine A per process
      9. Combine A*W per process
    */
    gpuTime = magma_sync_wtime(queue);
    magma_saxpy(conf.hiddenNeuron*conf.hiddenNeuron, 1.0, d_A, 1, d_Aproc, 1, queue);
    magma_sgemm(MagmaNoTrans, MagmaNoTrans, conf.hiddenNeuron, conf.classNum,
      conf.hiddenNeuron, 1.0 , d_A, conf.hiddenNeuron, d_W, conf.hiddenNeuron, 1.0,
      d_AWproc, conf.hiddenNeuron, queue);
    rt.combineW += (magma_sync_wtime(queue) - gpuTime);
  }

  /*
    Recombining all the output wieghts
    if K=1, then return the W else
      combine the A then combine the W
  */
  gpuTime = magma_sync_wtime(queue);
  if (totalSubCount == 1){
    magma_free(d_Wout);
    d_Wout = d_W;
  } else {
    MPI_Reduce(d_Aproc, d_Acombined, conf.hiddenNeuron*conf.hiddenNeuron,
      MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
    if (rank == ROOT){
      *alfa = (totalSubCount-1.0)/conf.alpha;
      addToDiagonal(cudaStream, d_Acombined, conf.hiddenNeuron, conf.hiddenNeuron,
        hiddenNeuron, hiddenNeuron, alfa);
      cudaStreamSynchronize(cudaStream);
    }
    MPI_Reduce(d_AWproc, d_AWcombined, conf.hiddenNeuron*conf.classNum,
      MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
    if (rank == ROOT){
      magma_sgetmatrix(*hiddenNeuron, *hiddenNeuron, d_Acombined, *hiddenNeuron, Acombined,
        *hiddenNeuron, queue);
      getPseudoInverse(queue, Acombined, d_AcombinedInv, *hiddenNeuron, *hiddenNeuron);
      magma_sgemm(MagmaNoTrans, MagmaNoTrans, conf.hiddenNeuron, conf.classNum, conf.hiddenNeuron,
        1.0 , d_AcombinedInv, conf.hiddenNeuron, d_AWcombined, conf.hiddenNeuron, 0., d_Wout,
        conf.hiddenNeuron, queue);
    }
  }
  rt.combineW += (magma_sync_wtime(queue) - gpuTime);
  std::printf("Rank %d: memory allocation : %.9lf seconds\n", rank, rt.memoryAllocation);
  std::printf("Rank %d: read data %.9lf seconds\n", rank, rt.readDataTime);
  std::printf("Rank %d: calculate H %.9lf seconds\n", rank, rt.maxH);
  std::printf("Rank %d: calculate A %.9lf seconds\n", rank, rt.maxA);
  std::printf("Rank %d: calculate W %.9lf seconds\n", rank, rt.maxW);
  std::printf("Rank %d: combining for Wout %.9lf seconds\n", rank, rt.combineW);

  double readTime, maxH, maxA, maxW, memAlloc, combineW;
  MPI_Reduce(&rt.memoryAllocation, &memAlloc, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&rt.readDataTime, &readTime, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&rt.maxH, &maxH, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&rt.maxA, &maxA, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&rt.maxW, &maxW, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);
  MPI_Reduce(&rt.combineW, &combineW, 1, MPI_DOUBLE, MPI_MAX, ROOT, MPI_COMM_WORLD);


  if (rank == ROOT){
    magma_sgetmatrix(conf.hiddenNeuron, conf.classNum, d_Wout, conf.hiddenNeuron, Wout,
      conf.hiddenNeuron, queue);
    write_smatrix(conf.wOutputFileName, Wout, conf.hiddenNeuron, conf.classNum);
    rt.subCount = conf.subCount;
    rt.np = numProcs;
    rt.row = conf.row;
    rt.col = conf.col;
    rt.hiddenNeuron = conf.hiddenNeuron;
    rt.readDataTime = readTime;
    rt.maxA = maxA;
    rt.maxH = maxH;
    rt.maxW = maxW;
    rt.memoryAllocation = memAlloc;
    rt.totalTime = rt.readDataTime + rt.maxH + rt.maxA + rt.maxW + rt.combineW;
    writeRunningTimeData(conf.runningTimeFileName, rt);
  }

  magma_queue_destroy(queue);
  magma_finalize();
  MPI_Finalize();
}
