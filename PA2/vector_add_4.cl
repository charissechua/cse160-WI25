__kernel void vectorAdd(__global const int *a, __global const int *b,
			            __global const int *c, __global const int *d,
                        __global int *result, const unsigned int size) {
                          //result[]
  //@@ Insert code to implement vector addition here
  //todo: result = a+b+c+d;
  int index = get_global_id(0);
  if (index < size){
    result[index]= a[index] + b[index] + c[index] + d[index];}
}
