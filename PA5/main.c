#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <libgen.h>
#include <string.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"
#include "img.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define KERNEL_PATH "kernel.cl"

#define COMPUTE_OUTUT_DIM(input_dim, kernel_size, stride) \
    ((input_dim - kernel_size) / stride + 1)

void OpenCLConvolution2D(Image *input0, Matrix *input1, Image *result, int stride)
{
    //? input0 is iag
    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(KERNEL_PATH); // Load kernel source

    // Device input and output buffers
    cl_mem device_a, device_b, device_c;

    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get the ID for the specified kind of device type.
    err = OclGetDeviceWithFallback(&device_id, OCL_DEVICE_TYPE);
    CHECK_ERR(err, "OclGetDeviceWithFallback");

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
    # if __APPLE__
        queue = clCreateCommandQueue(context, device_id, 0, &err);
    #else
        queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    #endif
        CHECK_ERR(err, "clCreateCommandQueueWithProperties");
        

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "convolution2D", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Allocate GPU memory here
    device_a = clCreateBuffer(context,
                              CL_MEM_READ_ONLY,
                              input0->shape[0] * input0->shape[1] * IMAGE_CHANNELS * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer device a");

    device_b = clCreateBuffer(context,
                              CL_MEM_READ_ONLY,
                              input1->shape[0] * input1->shape[1] * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer device b");

    device_c = clCreateBuffer(context,
                              CL_MEM_WRITE_ONLY,
                              result->shape[0] * result->shape[1] * IMAGE_CHANNELS * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer device c");


    //@@ Copy memory to the GPU here
    err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, sizeof(int) * input0->shape[0] * input0->shape[1] * IMAGE_CHANNELS, input0->data, 0, NULL, NULL);
    CHECK_ERR(err, "writing fgor input 1");

    err = clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, sizeof(int) * input1->shape[0] * input1->shape[1], input1->data, 0, NULL, NULL);
    CHECK_ERR(err, "writing fgor input 1");

    // Set the arguments to our compute kernel
    // __global float * inputData, __global float * outputData, __constant float * maskData,
    // int width, int height, int maskWidth,  int imageChannels
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_a);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_c);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_b);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &input0->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &input0->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 4");
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &input1->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 5");
    int imageChannels = IMAGE_CHANNELS;
    err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &imageChannels);
    CHECK_ERR(err, "clSetKernelArg 6");
    err |= clSetKernelArg(kernel, 7, sizeof(unsigned int), &stride);
    CHECK_ERR(err, "clSetKernelArg 7");

    // Compute the output dim 
    // @@ define local and global work sizes
    // Execute the OpenCL kernel on the list
    //? size of the entire output matrix which is the size of the input matrix 
    //? is gloabl work size supposed to have 3 dimensions?
    size_t global_work_size[3] = {result->shape[0], result->shape[1], result->shape[3]}; 
    // TODO local_work_size size_t local_work_size [2] = = {TILE_SIZE, TILE_SIZE}; 
    //@@ Launch the GPU Kernel here
    // Execute the OpenCL kernel on the array
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);   
    CHECK_ERR(err , "Kernel run");
    //@@ Copy the GPU memory back to the CPU here
    // Read the memory buffer output_mem_obj to the local variable result
    err = clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, result->shape[0] * result->shape[1] * IMAGE_CHANNELS * sizeof(int), result->data, 0, NULL, NULL);
    CHECK_ERR(err , "Kernel run");

    //@@ Free the GPU memory here
    // Release OpenCL resources
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_c);

    clReleaseKernel(kernel); 
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // get the dir from the input file
    int stride;
    char dir[256];
    strcpy(dir, dirname(strdup(input_file_a))); 

    // Host input and output vectors and sizes
    Image host_a, host_c, answer;
    Matrix host_b;
    
    cl_int err;

    err = LoadImgRaw(input_file_a, &host_a);
    CHECK_ERR(err, "LoadImg");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    // err = LoadImgTmp(input_file_c, &answer);
    err = LoadImgRaw(input_file_c, &answer);
    CHECK_ERR(err, "LoadImg");

    // Load stride
    err = LoadStride(dir, &stride);
    CHECK_ERR(err, "LoadStride");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer image
    rows = host_a.shape[0] - (host_b.shape[0] - 1); 
    cols = host_a.shape[1] - (host_b.shape[1] - 1); 
    
    // Allocate the memory for the target.
    host_c.shape[0] = rows;
    host_c.shape[1] = cols;
    host_c.data = (int *)malloc(sizeof(int) * host_c.shape[0] * host_c.shape[1] * IMAGE_CHANNELS);

    OpenCLConvolution2D(&host_a, &host_b, &host_c, stride);
    // printf("MY OUTPUT\n:");
    // PrintMatrix(&host_c);
    // printf("ANSWER \n:");
    // PrintMatrix(&answer);
    // Save the image
    SaveImg(input_file_d, &host_c);

    // Check the result of the convolution
    err = CheckImg(&answer, &host_c);
    CHECK_ERR(err, "CheckImg");

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}