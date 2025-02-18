__kernel void matrixMultiply(
    __global const int *A, __global const int *B, __global int *C,
    const unsigned int numARows, const unsigned int numAColumns,
    const unsigned int numBRows, const unsigned int numBColumns,
    const unsigned int numCRows, const unsigned int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //DOENST WORK UNLESS ITS A MULTIPLE OF THE TILE_SIZE
  int sum = 0; 

  // STEP 1: get all the indices 

  int x = get_local_id(0); //thread WITHIN the tile
  int y = get_local_id(1);
  int i = get_group_id(0); // which row of group ID it is 
  int j = get_group_id(1);
  int global_x = get_global_id(0); 
  int global_y = get_global_id(1);
  #define TILE_SIZE 16
  // STEP 2: initilaize local mem for A tile and B tile 
  __local int local_A[TILE_SIZE][TILE_SIZE]; //matches tile size, tile size = 4
  __local int local_B[TILE_SIZE][TILE_SIZE];

  int tile_iterations = (numAColumns + TILE_SIZE - 1)/TILE_SIZE; //get the ceiling when it isnt a multiple of tiles

  // need to translate from x and y index into the global memory index that it should be pulling from 

  // STEP 3: loop over number of tiles across A and B 
  // LOADING ALL DATA FROM GLOBAL TO LOCAL MEMORY 
  for (int k = 0; k < tile_iterations; k++ ){
      // STEP 4 (inside loop): load a tile from A and B into local memory 
      int row = get_global_id(0); 
      int col = get_global_id(1); 

      //A CALCULATIONS
      // i * TILE_SIZE get us to top edge of the tile, + x offsets us below the tile to get the row in it
      int A_row = (i * TILE_SIZE) + x; // gets NUMBER of rows 
      // k * TILE_SIZE gets us to the edge of the tile, + y offsets us within the tile 
      int A_col = k * TILE_SIZE + y; 

      //B CALCULATIONS 
      // k * TILE_SIZE gets us to the top edge of the tile because we have iterated down to it k times, each TILE_SIZE length rows, + offset x for local 
      int B_row =(k * TILE_SIZE) + x;
      // j * TILE_SIZE = edge of tile + y to move it over to appropriate col for local item 
      int B_col = (j * TILE_SIZE) + y; 


      //each thread responsible for loading specific piece of memory into local mem
      // Load tiles into local memory (with bounds checking)
        if (A_row < numARows && A_col < numAColumns) {
            local_A[x][y] = A[A_row * numAColumns + A_col];
        } else {
            local_A[x][y] = 0;  
        }

        if (B_row < numBRows && B_col < numBColumns) {
            local_B[x][y] = B[B_row * numBColumns + B_col];
        } else {
            local_B[x][y] = 0; 
        }

      // //gets one ELEMENT of A at a time, wanna offset by tile_col instead
      // local_A[x][y] = A[A_row * numAColumns + A_col]; 
      // //gets one ELEMENT of B at a time, wanna offset by tile_row instead
      // local_B[x][y] = B[B_row * numBColumns + B_col]; 

      barrier(CLK_LOCAL_MEM_FENCE);

      // STEP 5 (inside loop) calculate partial sum
      // loop over size of tile and do partial mult 
      //? doesnt this do only one partial sum? since there are 16 threads, it should only calculate for the corresponding one
      for (int w = 0; w < TILE_SIZE; w++){
        sum += local_A[x][w] * local_B[w][y];
      }
      barrier(CLK_LOCAL_MEM_FENCE); //so none of the threads go away 

  }


  // STEP 6 store the sum w global indices 
  if (global_x < numCRows && global_y < numCColumns) {
    C[global_x * numCColumns + global_y] = sum;
  }

}