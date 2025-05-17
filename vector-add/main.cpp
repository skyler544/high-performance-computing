#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if defined(_WIN32)
    #include <CL/cl.h>
#elif defined(__APPLE__)
    #include <OpenCL/opencl.h>
#else
    #include <CL/cl.h>
#endif

#include <fstream>
#include <string>
#include "cl_utils.hpp"

cl_device_id getDevice() {
    cl_int status;

    cl_uint numPlatforms = 0;
    checkStatus(clGetPlatformIDs(0, NULL, &numPlatforms));

    if (numPlatforms == 0) {
        printf("Error: No OpenCL platform available!\n");
        exit(EXIT_FAILURE);
    }

    cl_platform_id platform;
    checkStatus(clGetPlatformIDs(1, &platform, NULL));

    cl_uint numDevices = 0;
    checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));

    if (numDevices == 0) {
        printf("Error: No OpenCL device available for platform!\n");
        exit(EXIT_FAILURE);
    }

    cl_device_id device;
    checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

    return device;
}

cl_context getContext(cl_device_id device) {
    cl_int status;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkStatus(status);

    return context;
}

cl_command_queue getCommandQueue(cl_context context, cl_device_id device) {
    cl_int status;
    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
    checkStatus(status);

    return commandQueue;
}

cl_mem allocateBuffer(cl_context context, size_t dataSize, cl_mem_flags flag) {
    cl_int status;
    cl_mem buffer = clCreateBuffer(context, flag, dataSize, NULL, &status);
    checkStatus(status);

    return buffer;
}

void enqueueWriteBuffer(cl_command_queue commandQueue,
                        cl_mem buffer,
                        size_t dataSize,
                        const void* data) {
    checkStatus(
        clEnqueueWriteBuffer(commandQueue, buffer, CL_TRUE, 0, dataSize, data, 0, NULL, NULL));
}

cl_program initializeProgram(cl_context context, cl_device_id device) {
    cl_int status;

    // read the kernel source
    const char* kernelFileName = "kernel.cl";
    std::ifstream ifs(kernelFileName);
    if (!ifs.good()) {
        printf("Error: Could not open kernel with file name %s!\n", kernelFileName);
        exit(EXIT_FAILURE);
    }

    std::string kernelSource((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
    const char* kernelSourceArray = kernelSource.c_str();
    size_t programSize = kernelSource.length();

    // create the program
    cl_program program = clCreateProgramWithSource(
        context, 1, static_cast<const char**>(&kernelSourceArray), &programSize, &status);
    checkStatus(status);

    // build the program
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (status != CL_SUCCESS) {
        printCompilerError(program, device);
        exit(EXIT_FAILURE);
    }

    return program;
}

cl_kernel vectorAddKernel(cl_program program,
                          cl_mem bufferA,
                          cl_mem bufferB,
                          cl_uint numberOfElements) {
    cl_int status;
    cl_kernel kernel = clCreateKernel(program, "vector_add", &status);
    checkStatus(status);

    checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA));
    checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB));
    checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_uint), &numberOfElements));

    return kernel;
}

void enqueueKernel(cl_command_queue commandQueue, cl_kernel kernel, cl_uint numberOfElements) {
    size_t globalWorkSize = static_cast<size_t>(numberOfElements);
    checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, NULL, 0,
                                       NULL, NULL));
}

void enqueueReadBuffer(cl_command_queue commandQueue, cl_mem buffer, size_t dataSize, void* data) {
    checkStatus(
        clEnqueueReadBuffer(commandQueue, buffer, CL_TRUE, 0, dataSize, data, 0, NULL, NULL));
}

int main(int argc, char** argv) {
    // DATA SETUP
    // ------------------------------------------------
    // input and output arrays
    const cl_uint numberOfElements = 10;
    size_t dataSize = numberOfElements * sizeof(int32_t);
    // malloc returns a void * so we cast it to int32_t *
    int32_t* vectorA = static_cast<int32_t*>(malloc(dataSize));
    int32_t* vectorB = static_cast<int32_t*>(malloc(dataSize));

    for (unsigned int i = 0; i < numberOfElements; ++i) {
        vectorA[i] = static_cast<int32_t>(i);
    }

    // OPENCL SETUP
    // ------------------------------------------------
    cl_device_id device = getDevice();
    cl_context context = getContext(device);
    cl_command_queue commandQueue = getCommandQueue(context, device);

    cl_mem bufferA = allocateBuffer(context, dataSize, CL_MEM_READ_ONLY);
    cl_mem bufferB = allocateBuffer(context, dataSize, CL_MEM_WRITE_ONLY);
    enqueueWriteBuffer(commandQueue, bufferA, dataSize, vectorA);

    cl_program program = initializeProgram(context, device);
    cl_kernel kernel = vectorAddKernel(program, bufferA, bufferB, numberOfElements);

    // RUN PROGRAM
    // ------------------------------------------------
    enqueueKernel(commandQueue, kernel, numberOfElements);
    enqueueReadBuffer(commandQueue, bufferB, dataSize, vectorB);
    printVector(vectorA, numberOfElements, "Input A");
    printVector(vectorB, numberOfElements, "Output B");

    // CLEANUP
    // ------------------------------------------------
    // release allocated resources
    free(vectorB);
    free(vectorA);

    // release opencl objects
    checkStatus(clReleaseKernel(kernel));
    checkStatus(clReleaseProgram(program));
    checkStatus(clReleaseMemObject(bufferB));
    checkStatus(clReleaseMemObject(bufferA));
    checkStatus(clReleaseCommandQueue(commandQueue));
    checkStatus(clReleaseContext(context));

    exit(EXIT_SUCCESS);
}
