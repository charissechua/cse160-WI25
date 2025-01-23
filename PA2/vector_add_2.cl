__kernel void vectorAdd(__global const int *a, __global const int *b,
                        __global int *result, const unsigned int size) {
  //@@ Insert code to implement vector addition here
  //? kernel computes for a single index in the matrix? how do we know which index to put it in?
  int index = get_global_id(0);

  if (index < size){
    result[index]= a[index] + b[index];}
}
