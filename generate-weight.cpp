#include <stdio.h>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

#include "helper_lib.h"
#include "magma_v2.h"
#include "magma_lapack.h"

#define ROOT 0

using namespace std;

int main(int argc, char **argv){
  magma_init();
  magma_queue_t queue = NULL;
  magma_queue_create(0, &queue);
  magma_int_t err;

  int row = stoi(argv[2]); // colomn of X data
  int col = stoi(argv[3]); //hidden neuron
  string dataName = argv[1];
  string fileName = "weight/"+dataName+"/w-in-"+argv[3]+".bin";
  ofstream wFile(fileName, ios::trunc | ios::binary);

  float *Winp;
  magma_smalloc_cpu(&Winp, row*col);
  magma_int_t ione=2;
  magma_int_t ISEED[4]={0, 1, 2, 3};
  magma_int_t wSize = row*col;
  lapackf77_slarnv(&ione, ISEED, &wSize, Winp);
  write_smatrix(fileName, Winp, row, col);

  magma_queue_destroy(queue);
  magma_finalize();
}
