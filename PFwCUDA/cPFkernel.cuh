#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void PF_iteration_kernel(cudaPitchedPtr mPtr, cudaExtent mExt, dim3 matrix_dimensions, 
									double src, dim3 srcloc, bool * wallLoc, float * WWall, float * W,
									cudaPitchedPtr nmPtr);
void callerblahblah(void);
void cPFcaller(unsigned int num_iterations, float * m_ptr);
void cPFaddWallLocation(int x, int y, bool val);
void cPFdelete(void);
void cPFinit(double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[]);