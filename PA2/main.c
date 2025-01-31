#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kernel.h"
#include "matrix.h"
#include "device.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define VECTOR_ADD_2_KERNEL_PATH "vector_add_2.cl"
#define VECTOR_ADD_4_KERNEL_PATH "vector_add_4.cl"

void initializeOpenCL(cl_device_id* device_id, cl_context* context, cl_command_queue* queue) {
    cl_int err;

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;
    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get ID for first device on first platform
    *device_id = platforms[0].devices[0].device_id;

    // Create a context
    *context = clCreateContext(0, 1, device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
# if __APPLE__
    *queue = clCreateCommandQueue(*context, *device_id, 0, &err);
# else
    *queue = clCreateCommandQueueWithProperties(*context, *device_id, 0, &err);
# endif
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");
}

void callVectorAdd2Kernel(Matrix* a, Matrix* b, Matrix* out, cl_context* context, cl_command_queue* queue) {
    //printf("in vectoradd2kernel\n");
    //a, b, out are all cpu 
    //everything else that we allocate are on gpu 

    // OpenCL objects
    cl_program program;                 // program
    cl_kernel kernel;         // kernel

    // OpenCL setup variables
    size_t global_item_size, local_item_size;
    cl_int err;

    // Device input and output vectors
    cl_mem device_input_1, device_input_2, device_output;

    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(VECTOR_ADD_2_KERNEL_PATH);

    // Create the program from the source buffer
    program = clCreateProgramWithSource(*context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");
    
    kernel = clCreateKernel(program, "vectorAdd", &err);
    CHECK_ERR(err, "clCreateKernel");

    // Allocate GPU memory
    // Create memory buffers for input and output vectors
    device_input_1 = clCreateBuffer(*context,
                              CL_MEM_READ_ONLY,
                              a->shape[0] * a->shape[1] * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer a");

    //like malloc which allocates space for us
    device_input_2 = clCreateBuffer(*context,
                              CL_MEM_READ_ONLY,
                              b->shape[0] * b->shape[1] * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer b");

    device_output = clCreateBuffer(*context,
                              CL_MEM_WRITE_ONLY,
                              out->shape[0] * out->shape[1] * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer out");

    //@@ Copy memory to the GPU here
    //todo: 1. copy the input data to the device, using device_input_1 and device_input_2
    //does each device_input_1 contain one matrix?
    //printf ("======ENQUEUEWRITEBUFFER===== \n");
    clEnqueueWriteBuffer(*queue, device_input_1, CL_FALSE, 0, sizeof(int) * a->shape[0], a->data, 0, NULL, NULL);
    clEnqueueWriteBuffer(*queue, device_input_2, CL_FALSE, 0, sizeof(int) * b->shape[0], b->data, 0, NULL, NULL);


    //@@ define local and global work sizes
    //todo:defines size of the ND range kernel, 1 dimensional and == to size of vector
    // size as in memory? or size as in dimensions? for the matrix --> number of elements not the memory size 
    // do we need size_b as well for that matrix? --> same as size_b
    unsigned int size_a = a->shape[0]; 

    // Set the arguments to the kernel alr done for us
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_input_1);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_input_2);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_output);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &size_a);
    CHECK_ERR(err, "clSetKernelArg 3");

    //@@ Launch the GPU Kernel here
    // todo: 1. set the dimensions for the NDRange
    // todo: 2. enqueue the kernel
    // size_t global_work_size[2] if it was a 2d work group and then assign each element 
    // global_work_size[0] = 10, global_work_size[1] = 2 so num work items = 20
    //device output buffer has the solution from kernel code 
    //calls the kernel and runs the func

    global_item_size = a->shape[0]; 
    //printf ("======ND_RANGE_KERNEL===== \n");
    clEnqueueNDRangeKernel(*queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);   

    //@@ Copy the GPU memory back to the CPU here
    //device_output = GPU and out = CPU
    // todo: output is stored device_output copy back from gpu device memory using clenqueue read buffer
    clEnqueueReadBuffer(*queue, device_output, CL_TRUE, 0, a->shape[0] * sizeof(int), out->data, 0, NULL, NULL);

    //@@ Free the GPU memory here
    // todo: 1. free the gpu
    //? r the steps for add2kernel and add4kernel the same for set up?

    //printf ("======FREEING DEVICE INPUTS/OUTPUTS BUFFERS===== \n");
    clReleaseMemObject(device_input_1);
    clReleaseMemObject(device_input_2);
    clReleaseMemObject(device_output);

   // printf ("======FREEING KERNEL===== \n");

    // Release Host Memory
    free(kernel_source);
}

void part1(Matrix* host_input_1, Matrix* host_input_2, Matrix* host_input_3, Matrix* host_input_4, Matrix* host_output, Matrix* answer, const char* output_file) {
    // Start of program one

    // OpenCL objects
    //CREATING CONTEXT & COMMAND QUEUE
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue

    initializeOpenCL(&device_id, &context, &queue);
    //printf("STARTING VECTORADD2KERNEL CALLS\n");

    callVectorAdd2Kernel(host_input_1, host_input_2, host_output, &context, &queue);
   // printf("CALL 1 DONE\n");

    callVectorAdd2Kernel(host_output, host_input_3, host_output, &context, &queue);
    //printf("CALL 2 DONE\n");

    callVectorAdd2Kernel(host_output, host_input_4, host_output, &context, &queue);
    //printf("FINISHED VECTORADD2KERNEL CALLS\n");

    // Prints the results
    // for (unsigned int i = 0; i < host_output.shape[0] * host_output.shape[1]; i++)
    // {
    //     printf("C[%u]: %d == %d\n", i, host_output.data[i], answer.data[i]);
    // }

    // Check whether the answer matches the output

    
    CheckMatrix(answer, host_output);
    //printf("saving matrix\n");
    SaveMatrix(output_file, host_output);

    //@@ Release OpenCL objects here
    //? do we also have to release the inputs and outputs?
   // printf("RELEASING COMMAND QUEUE\n");
    clReleaseContext(context); 
    clReleaseCommandQueue(queue); 
    //printf("RELEASING CONTEXT\n");
   

}

void callVectorAdd4Kernel(Matrix* a, Matrix* b, Matrix* c, Matrix* d, Matrix* out, cl_context* context, cl_command_queue* queue) {
    // OpenCL objects
    cl_program program;       // program
    cl_kernel kernel;         // kernel

    // OpenCL setup variables
    size_t global_item_size, local_item_size;
    cl_int err;

    // Device input and output vectors
    cl_mem device_input_1, device_input_2, device_input_3, device_input_4, device_output;

    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(VECTOR_ADD_4_KERNEL_PATH);

    // Create the program from the source buffer
    program = clCreateProgramWithSource(*context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");
    
    kernel = clCreateKernel(program, "vectorAdd", &err);
    CHECK_ERR(err, "clCreateKernel");

    // Allocate GPU memory
    // Create memory buffers for input and output vectors
    device_input_1 = clCreateBuffer(*context,
                              CL_MEM_READ_ONLY,
                              a->shape[0] * a->shape[1] * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer a");

    device_input_2 = clCreateBuffer(*context,
                              CL_MEM_READ_ONLY,
                              b->shape[0] * b->shape[1] * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer b");

    device_input_3 = clCreateBuffer(*context,
                              CL_MEM_READ_ONLY,
                              c->shape[0] * c->shape[1] * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer c");

    device_input_4 = clCreateBuffer(*context,
                              CL_MEM_READ_ONLY,
                              d->shape[0] * d->shape[1] * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer d");

    device_output = clCreateBuffer(*context,
                              CL_MEM_WRITE_ONLY,
                              out->shape[0] * out->shape[1] * sizeof(int),
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer out");

    //@@ Copy memory to the GPU here
    clEnqueueWriteBuffer(*queue, device_input_1, CL_FALSE, 0, sizeof(int) * a->shape[0], a->data, 0, NULL, NULL);
    clEnqueueWriteBuffer(*queue, device_input_2, CL_FALSE, 0, sizeof(int) * b->shape[0], b->data, 0, NULL, NULL);
    clEnqueueWriteBuffer(*queue, device_input_3, CL_FALSE, 0, sizeof(int) * c->shape[0], c->data, 0, NULL, NULL);
    clEnqueueWriteBuffer(*queue, device_input_4, CL_FALSE, 0, sizeof(int) * d->shape[0], d->data, 0, NULL, NULL);


    //@@ define local and global work sizes
    unsigned int size_a = a->shape[0];// @@ replace this with sizeof(a)

    // Set the arguments to the kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_input_1);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_input_2);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_input_3);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &device_input_4);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &device_output);
    CHECK_ERR(err, "clSetKernelArg 4");
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &size_a);
    CHECK_ERR(err, "clSetKernelArg 5");

    //@@ Launch the GPU Kernel here
    global_item_size = a->shape[0];
    clEnqueueNDRangeKernel(*queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);   

    //@@ Copy the GPU memory back to the CPU here
    clEnqueueReadBuffer(*queue, device_output, CL_TRUE, 0, a->shape[0] * sizeof(int), out->data, 0, NULL, NULL);

    //@@ Free the GPU memory here
    clReleaseMemObject(device_input_1);
    clReleaseMemObject(device_input_2);
    clReleaseMemObject(device_input_3);
    clReleaseMemObject(device_input_4);
    clReleaseMemObject(device_output);

    // Release Host Memory
    free(kernel_source);
}

void part2(Matrix* host_input_1, Matrix* host_input_2, Matrix* host_input_3, Matrix* host_input_4, Matrix* host_output, Matrix* answer, const char* output_file) {
    // Start of program two

    // OpenCL objects
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue

    initializeOpenCL(&device_id, &context, &queue);

    callVectorAdd4Kernel(host_input_1, host_input_2, host_input_3, host_input_4, host_output, &context, &queue);

    // Prints the results
    // for (unsigned int i = 0; i < host_output.shape[0] * host_output.shape[1]; i++)
    // {
    //     printf("C[%u]: %d == %d\n", i, host_output.data[i], answer.data[i]);
    // }

    // Check whether the answer matches the output
    CheckMatrix(answer, host_output);
    SaveMatrix(output_file, host_output);

    //@@ Release OpenCL objects here
    //clReleaseCommandQueue(queue); 
    clReleaseContext(context);    
}

int main(int argc, char *argv[])
{
    if (argc != 8)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <input_file_2> <input_file_3> <answer_file> <output_file_program_1> <output_file_program_2>\n", argv[0]);
        return -1;
    }

    const char *input_array_1_file = argv[1];
    const char *input_array_2_file = argv[2];
    const char *input_array_3_file = argv[3];
    const char *input_array_4_file = argv[4];
    const char *answer_file = argv[5];
    const char *program_1_output_file = argv[6];
    const char *program_2_output_file = argv[7];

    // Host input and output vectors
    Matrix host_input_1, host_input_2, host_input_3, host_input_4, host_output, answer;

    // OpenCL setup variables
    cl_int err;

    // Load input matrix from file and check for errors
    err = LoadMatrix(input_array_1_file, &host_input_1);
    CHECK_ERR(err, "LoadMatrix");
    err = LoadMatrix(input_array_2_file, &host_input_2);
    CHECK_ERR(err, "LoadMatrix");
    err = LoadMatrix(input_array_3_file, &host_input_3);
    CHECK_ERR(err, "LoadMatrix");
    err = LoadMatrix(input_array_4_file, &host_input_4);
    CHECK_ERR(err, "LoadMatrix");
    err = LoadMatrix(answer_file, &answer);
    CHECK_ERR(err, "LoadMatrix");

    // Allocate the memory for the output
    host_output.shape[0] = host_input_1.shape[0];
    host_output.shape[1] = host_input_1.shape[1];
    host_output.data = (int *)calloc(sizeof(int), host_output.shape[0] * host_output.shape[1]);

    // Time measurement variables
    clock_t start, end;
    double cpu_time_used;

    // =================================================================
    printf("==============Starting Program 1==============\n");
    start = clock();

    part1(&host_input_1, &host_input_2, &host_input_3, &host_input_4, &host_output, &answer, program_1_output_file);
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds

    printf("Execution time: %.2fms\n", cpu_time_used);
    //printf("==============Finished Program 1==============\n");

    // Cleanup and prepare for second program.
   // printf("==============about to free data==============\n");

    free(host_output.data);
    host_output.data = (int *)calloc(sizeof(int), host_output.shape[0] * host_output.shape[1]);

    // =================================================================
    printf("==============Starting Program 2==============\n");
    start = clock();

    part2(&host_input_1, &host_input_2, &host_input_3, &host_input_4, &host_output, &answer, program_2_output_file);
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
    printf("Execution time: %.2fms\n", cpu_time_used);
    printf("==============Finished Program 2==============\n");



    // Release host memory
    free(host_input_1.data);
    free(host_input_2.data);
    free(host_input_3.data);
    free(host_input_4.data);
    free(host_output.data);
    free(answer.data);

    return 0;
}
