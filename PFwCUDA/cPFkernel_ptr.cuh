#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void PF_ptr_flow(cudaPitchedPtr mPtr, cudaExtent mExt, dim3 matrix_dimensions, 
									double src, dim3 srcloc, bool * wallLoc, float * WWall, float * W,
									cudaPitchedPtr nmPtr);
__global__ void PF_ptr_copy(cudaPitchedPtr mPtr, cudaPitchedPtr nmPtr, cudaExtent mExt, dim3 matdim);
__global__ void PF_setup_globals(size_t ptrpitch, cudaExtent mExt, dim3 matdim);
void cPFcaller(unsigned int num_iterations, float * &m_ptr);
void cPFaddWallLocation(int x, int y, bool val);
void cPFdelete(void);
void cPFinit(float matrixFlow[][4], float matrixWall[][4], float in_sourceLoc[]);
