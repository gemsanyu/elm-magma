the configs will be read as arguments

data-name train-data-row-size data-col-size class-size hidden-neuron-size alfa subdata-count running-time-filename

TO RUN 1 process per node with 1 sub data count, example:
mpirun --bind-to none -pernode ./main mnist 60000 784 10 1100 0.1 1 pc-running-time.csv

TO TEST 1 process per node with 2 sub data count, example:
mpirun --bind-to none -pernode ./test mnist 10000 784 10 1100 0.1 2 pc-test.csv

1. read inputs and make sure its transposed (column major)
2. do computation
  a. make sure to save much memory
  b. make sure to save much time (less memory handling)
3. check result
