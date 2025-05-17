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

    // OPENCL PLATFORM AND DEVICE SETUP
    // ------------------------------------------------
    // used for checking error status of api calls
    cl_int status;

    // the first call to clGetPlatformIDs is used to check if we have any
    // platforms at all. the second call is used to retrieve the platform
    // itself. why is this not two different functions?

    // retrieve the number of platforms
    cl_uint numPlatforms = 0;
    checkStatus(clGetPlatformIDs(0, NULL, &numPlatforms));

    if (numPlatforms == 0) {
        printf("Error: No OpenCL platform available!\n");
        exit(EXIT_FAILURE);
    }

    // select the FIRST platform
    // we pass our platform by reference so that it can be mutated?! why is this
    // not a function that just returns the platform so we can set it directly
    // like any normal person would do?
    // cl_platform_id platform = clGetPlatform(1); // so easy!
    cl_platform_id platform;
    checkStatus(clGetPlatformIDs(1, &platform, NULL));

    // retrieve the number of devices
    // same thing here with clGetDeviceIDs
    cl_uint numDevices = 0;
    checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));

    if (numDevices == 0) {
        printf("Error: No OpenCL device available for platform!\n");
        exit(EXIT_FAILURE);
    }

    // select the device
    cl_device_id device;
    checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

    // OPENCL CONTEXT AND COMMAND QUEUE SETUP
    // ------------------------------------------------
    // create context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    checkStatus(status);

    // create command queue
    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
    checkStatus(status);

    // allocate two input and one output buffer for the three vectors
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
    checkStatus(status);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status);
    checkStatus(status);

    // write data from the input vectors to the buffers
    checkStatus(
        clEnqueueWriteBuffer(commandQueue, bufferA, CL_TRUE, 0, dataSize, vectorA, 0, NULL, NULL));

    // OPENCL KERNEL SETUP
    // ------------------------------------------------
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

    // create the vector addition kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &status);
    checkStatus(status);

    // set the kernel arguments
    checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA));
    checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB));
    checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_uint), &numberOfElements));

    // DEVICE INFORMATION
    // ------------------------------------------------
    // output device capabilities
    size_t maxWorkGroupSize;
    checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                                &maxWorkGroupSize, NULL));
    printf("Device Capabilities: Max work items in single group: %zu\n", maxWorkGroupSize);

    cl_uint maxWorkItemDimensions;
    checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint),
                                &maxWorkItemDimensions, NULL));
    printf("Device Capabilities: Max work item dimensions: %u\n", maxWorkItemDimensions);

    size_t* maxWorkItemSizes = static_cast<size_t*>(malloc(maxWorkItemDimensions * sizeof(size_t)));
    checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                                maxWorkItemDimensions * sizeof(size_t), maxWorkItemSizes, NULL));
    printf("Device Capabilities: Max work items in group per dimension:");
    for (cl_uint i = 0; i < maxWorkItemDimensions; ++i)
        printf(" %u:%zu", i, maxWorkItemSizes[i]);
    printf("\n");
    free(maxWorkItemSizes);

    // RUN PROGRAM
    // ------------------------------------------------
    // execute the kernel
    // ndrange capabilites only need to be checked when we specify a local work
    // group size manually in our case we provide NULL as local work group size,
    // which means groups get formed automatically
    size_t globalWorkSize = static_cast<size_t>(numberOfElements);
    checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, NULL, 0,
                                       NULL, NULL));

    // read the device output buffer to the host output array
    checkStatus(
        clEnqueueReadBuffer(commandQueue, bufferB, CL_TRUE, 0, dataSize, vectorB, 0, NULL, NULL));

    // output result
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
