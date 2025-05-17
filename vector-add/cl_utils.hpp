#ifndef CL_UTILS_H
#define CL_UTILS_H

#include <CL/cl.h>
#include <cstdint>
#include <string>

std::string cl_errorstring(cl_int err);
void checkStatus(cl_int err);
void printCompilerError(cl_program program, cl_device_id device);
void printVector(int32_t* vector, unsigned int numberOfElements, const char* label);

#endif
