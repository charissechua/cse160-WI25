__kernel void matrixMultiply(
    __global const int *A, __global const int *B, __global int *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Compute C = A^T B 

  //computes it for one single output in column c
  int sum = 0; 
  int i = get_global_id(0); //x of output c
  int j = get_global_id(1); //y of output c
  
  if (i < numCRows && j < numCColumns) {

    for (int k = 0; k < numARows; k ++){
      int a = A[k * numAColumns + i];
      int b = B[k * numBColumns + j];
      sum += (a * b);
    }
      C[i * numCColumns + j] = sum;

  }  
}
