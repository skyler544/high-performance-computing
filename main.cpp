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

// AUXILIARY FUNCTIONS
// ----------------------------------------------------
std::string cl_errorstring(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return std::string("Success");
        case CL_DEVICE_NOT_FOUND: return std::string("Device not found");
        case CL_DEVICE_NOT_AVAILABLE: return std::string("Device not available");
        case CL_COMPILER_NOT_AVAILABLE: return std::string("Compiler not available");
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return std::string("Memory object allocation failure");
        case CL_OUT_OF_RESOURCES: return std::string("Out of resources");
        case CL_OUT_OF_HOST_MEMORY: return std::string("Out of host memory");
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return std::string("Profiling information not available");
        case CL_MEM_COPY_OVERLAP: return std::string("Memory copy overlap");
        case CL_IMAGE_FORMAT_MISMATCH: return std::string("Image format mismatch");
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return std::string("Image format not supported");
        case CL_BUILD_PROGRAM_FAILURE: return std::string("Program build failure");
        case CL_MAP_FAILURE: return std::string("Map failure");
        case CL_MISALIGNED_SUB_BUFFER_OFFSET: return std::string("Misaligned sub buffer offset");
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return std::string("Exec status error for events in wait list");
        case CL_INVALID_VALUE: return std::string("Invalid value");
        case CL_INVALID_DEVICE_TYPE: return std::string("Invalid device type");
        case CL_INVALID_PLATFORM: return std::string("Invalid platform");
        case CL_INVALID_DEVICE: return std::string("Invalid device");
        case CL_INVALID_CONTEXT: return std::string("Invalid context");
        case CL_INVALID_QUEUE_PROPERTIES: return std::string("Invalid queue properties");
        case CL_INVALID_COMMAND_QUEUE: return std::string("Invalid command queue");
        case CL_INVALID_HOST_PTR: return std::string("Invalid host pointer");
        case CL_INVALID_MEM_OBJECT: return std::string("Invalid memory object");
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return std::string("Invalid image format descriptor");
        case CL_INVALID_IMAGE_SIZE: return std::string("Invalid image size");
        case CL_INVALID_SAMPLER: return std::string("Invalid sampler");
        case CL_INVALID_BINARY: return std::string("Invalid binary");
        case CL_INVALID_BUILD_OPTIONS: return std::string("Invalid build options");
        case CL_INVALID_PROGRAM: return std::string("Invalid program");
        case CL_INVALID_PROGRAM_EXECUTABLE: return std::string("Invalid program executable");
        case CL_INVALID_KERNEL_NAME: return std::string("Invalid kernel name");
        case CL_INVALID_KERNEL_DEFINITION: return std::string("Invalid kernel definition");
        case CL_INVALID_KERNEL: return std::string("Invalid kernel");
        case CL_INVALID_ARG_INDEX: return std::string("Invalid argument index");
        case CL_INVALID_ARG_VALUE: return std::string("Invalid argument value");
        case CL_INVALID_ARG_SIZE: return std::string("Invalid argument size");
        case CL_INVALID_KERNEL_ARGS: return std::string("Invalid kernel arguments");
        case CL_INVALID_WORK_DIMENSION: return std::string("Invalid work dimension");
        case CL_INVALID_WORK_GROUP_SIZE: return std::string("Invalid work group size");
        case CL_INVALID_WORK_ITEM_SIZE: return std::string("Invalid work item size");
        case CL_INVALID_GLOBAL_OFFSET: return std::string("Invalid global offset");
        case CL_INVALID_EVENT_WAIT_LIST: return std::string("Invalid event wait list");
        case CL_INVALID_EVENT: return std::string("Invalid event");
        case CL_INVALID_OPERATION: return std::string("Invalid operation");
        case CL_INVALID_GL_OBJECT: return std::string("Invalid OpenGL object");
        case CL_INVALID_BUFFER_SIZE: return std::string("Invalid buffer size");
        case CL_INVALID_MIP_LEVEL: return std::string("Invalid mip-map level");
        case CL_INVALID_GLOBAL_WORK_SIZE: return std::string("Invalid gloal work size");
        case CL_INVALID_PROPERTY: return std::string("Invalid property");
        default: return std::string("Unknown error code");
    }
}

void checkStatus(cl_int err) {
    if (err != CL_SUCCESS) {
        printf("OpenCL Error: %s \n", cl_errorstring(err).c_str());
        exit(EXIT_FAILURE);
    }
}

void printCompilerError(cl_program program, cl_device_id device) {
    cl_int status;
    size_t logSize;
    char* log;

    // get log size
    status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    checkStatus(status);

    // allocate space for log
    log = static_cast<char*>(malloc(logSize));
    if (!log) {
        exit(EXIT_FAILURE);
    }

    // read the log
    status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
    checkStatus(status);

    // print the log
    printf("Build Error: %s\n", log);
}

void printVector(int32_t* vector, unsigned int numberOfElements, const char* label) {
    printf("%s:\n", label);

    for (unsigned int i = 0; i < numberOfElements; ++i) {
        printf("%d ", vector[i]);
    }

    printf("\n");
}

int main(int argc, char** argv) {
    // DATA SETUP
    // ------------------------------------------------
    // input and output arrays
    const unsigned int numberOfElements = 10;
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
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &status);
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

    printf("Kernel created successfully\n");
    // set the kernel arguments
    checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA));
    printf("Kernel arg 1 set successfully\n");
    checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB));
    printf("Kernel arg 2 set successfully\n");
    checkStatus(clSetKernelArg(kernel, 2, sizeof(cl_mem), &numberOfElements));
    printf("Kernel arg 2 set successfully\n");

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
