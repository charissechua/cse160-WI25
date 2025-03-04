#include <cmath>
#include <iostream>

#include "kernel.h"
#include "device.h"

#include "opencl-new-forward.h"

#define TILE_WIDTH 16

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d.\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }
	
void OpenCLInterface::conv_forward_opencl_prolog(const float *host_y, 
    const float *host_x, const float *host_k, cl_mem *device_y, 
    cl_mem *device_x, cl_mem *device_k, const int B, const int M, const int C, 
    const int H, const int W, const int K)
    //B = batch size, M = # output feature maps, H = Height input feature maps, W = width input feature maps, K = mask
{
    //@@ Allocate OpenCL memory here
    // Create memory buffers for input and output vectors
    // Do not create your own device/context/queue. 
    // Use this->opencl->[program, kernel, queue, context]
    // OpenCL (common for entire NN)
    //      class is defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl.h
    //      methods defined here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6%2Fsrc%2Flayer%2Fcustom%opencl.cc
    //      created and passed into the network here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/m2.cc
    //      it's pointer is kept in OpenCLInterface (THIS) class here: https://github.com/KastnerRG/cse160-WI25/blob/main/PA6/src/layer/custom/opencl-new-forward.h
    /**
        // Data transfer CPU to GPU
        openclInterface.conv_forward_opencl_prolog(y, x, k, &y_d, &x_d, &k_d, B, M, C, height_in, width_in, K);
    */
    cl_int err;
    //input data
    size_t size_x = B * C * H * W * sizeof(float);
    *device_x = clCreateBuffer(this->opencl->context,
                              CL_MEM_READ_ONLY,
                              size_x,
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer device_x");
    //kernel size
    size_t size_k = M * C * K * K * sizeof(float);
    *device_k = clCreateBuffer(this->opencl->context,
                              CL_MEM_READ_ONLY,
                              size_k,
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer device_k");
    //output data 
    int H_out = H - K + 1; 
    int W_out = W - K + 1;
    size_t size_y = B * M * H_out * W_out * sizeof(float);
    *device_y = clCreateBuffer(this->opencl->context,
                              CL_MEM_WRITE_ONLY,
                              size_y,
                              NULL,
                              &err);
    CHECK_ERR(err, "clCreateBuffer device_y");

    //@@ Copy memory to the OpenCL here
    // Copy input vectors to memory buffers
    err = clEnqueueWriteBuffer(this->opencl->queue, *device_x, CL_TRUE, 0, size_x, host_x, 0, NULL, NULL);
    CHECK_ERR(err, "writing for host_x");
    err = clEnqueueWriteBuffer(this->opencl->queue, *device_k, CL_TRUE, 0, size_k, host_k, 0, NULL, NULL);
    CHECK_ERR(err, "writing for kernel");
}

void OpenCLInterface::conv_forward_opencl(cl_mem device_y, const cl_mem device_x, 
    const cl_mem device_k, const int B, const int M, const int C, const int H, 
    const int W, const int K)
{

    //__global float *y, __constant float *x, __constant float *k,
    // const int B, const int M, const int C, const int H, const int W, const int K)
    // Set the arguments to our compute kernel
    //
    // Do not create your own device/context/queue.
    // Use this->opencl->[program, kernel, queue, context]

    //@@ Set the kernel dimensions and call the kernel
    //__global float *y, __constant float *x,
    // __constant float *k, const int B, const int M, const int C, const int H, 
    // const int W, const int K
    cl_int err;
    err = clSetKernelArg(this->opencl->kernel, 0, sizeof(cl_mem), &device_y);
    CHECK_ERR(err, "clSetKernelArg 0: device_y");
    err |= clSetKernelArg(this->opencl->kernel, 1, sizeof(cl_mem), &device_x);
    CHECK_ERR(err, "clSetKernelArg 1: device_x");
    err |= clSetKernelArg(this->opencl->kernel, 2, sizeof(cl_mem), &device_k);
    CHECK_ERR(err, "clSetKernelArg 2: device_k");
    err |= clSetKernelArg(this->opencl->kernel, 3, sizeof(int), &B);
    CHECK_ERR(err, "clSetKernelArg 3: B");
    err |= clSetKernelArg(this->opencl->kernel, 4, sizeof(int), &M);
    CHECK_ERR(err, "clSetKernelArg 4: M");
    err |= clSetKernelArg(this->opencl->kernel, 5, sizeof(int), &C);
    CHECK_ERR(err, "clSetKernelArg 5: C");
    err |= clSetKernelArg(this->opencl->kernel, 6, sizeof(int), &H);
    CHECK_ERR(err, "clSetKernelArg 6: H");
    err |= clSetKernelArg(this->opencl->kernel, 7, sizeof(int), &W);
    CHECK_ERR(err, "clSetKernelArg 7: W");
    err |= clSetKernelArg(this->opencl->kernel, 8, sizeof(int), &K);
    CHECK_ERR(err, "clSetKernelArg 8: K");

    //@@ Launch the OpenCL Kernel here
    // Execute the OpenCL kernel on the array
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    size_t global_work_size[3] = { (size_t)W_out, (size_t)H_out, (size_t)(B * M) };
    err = clEnqueueNDRangeKernel(this->opencl->queue, this->opencl->kernel,
                                3, NULL, global_work_size, NULL, 0, NULL, NULL);
    CHECK_ERR(err, "Kernel run");

}

void OpenCLInterface::conv_forward_opencl_epilog(float *host_y, cl_mem device_y, 
    cl_mem device_x, cl_mem device_k, const int B, const int M, const int C, 
    const int H, const int W, const int K)
{
    //@@ Copy the output back to host
    cl_int err;
    int H_out = H - K + 1; 
    int W_out = W - K + 1;
    err = clEnqueueReadBuffer(this->opencl->queue, device_y, CL_TRUE, 0, B * M * H_out * W_out * sizeof(float), host_y, 0, NULL, NULL);
    CHECK_ERR(err , "Kernel run");
    // Read the memory buffer output_mem_obj to the local variable result
    //
    // Do not create your own device/context/queue.
    // Use this->opencl->[program, kernel, queue, context]

    //@@ Free the OpenCL memory here
    // Release OpenCL resources

    clReleaseMemObject(device_y);
    clReleaseMemObject(device_x);
    clReleaseMemObject(device_k);
}
