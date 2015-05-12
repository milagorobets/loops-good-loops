#include "common.h"
#include "gpuPF_ptr.h"
#include "cPFkernel_ptr.cuh"
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <stdlib.h>
#include <cstring>

// default constructor
GPU_PTR::GPU_PTR()
{
	// Just create an instance, init later
	m_ptr = NULL;
}

void setupDisplay(void)
{
	// Call a kernel function to setup the OpenGL display
}

double GPU_PTR::get_M0(int x, int y)
{
	//return m0[x][y];
	// use pointer to access m0 data from GPU
	if (m_ptr != NULL)
	{
		return m_ptr[y*MATRIX_DIM+x]; // z = 0, so doesn't matter right now
	}
	else
	{
		return 0;
	}
}

void GPU_PTR::setMatrixWallLoc(int x, int y, int val)
{
	cPFaddWallLocation(x,y,val);
}

void GPU_PTR::cpuAlgoPixelFlow_init(float matrixFlow[][4], float matrixWall[][4], float in_sourceLoc[])
{
	cPFinit(matrixFlow, matrixWall, in_sourceLoc);
}

void GPU_PTR::cpuAlgoPixelFlow_delete()
{
	cPFdelete(); // Delete some malloc'ed memory
}

void GPU_PTR::cpuAlgoPixelFlow(unsigned int num_iterations)
{
	//cPFcaller(num_iterations, m_ptr);
	cPFcaller_display(num_iterations, m_ptr);
}

void GPU_PTR::cpuAlgoPixelFlow_updateSource(int t)
{
	// Don't need to update it here, kernels will do it
}

void GPU_PTR::cpuAlgoPixelFlow_nextStep(void)
{
	// Nothing to do here, see cPFkernel.cu instead!
}

