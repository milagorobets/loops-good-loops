#ifndef H_KERNEL // Header guards
#define H_KERNEL // Header guards

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

#endif // Header guards