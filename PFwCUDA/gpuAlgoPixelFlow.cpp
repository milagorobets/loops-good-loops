/*
 * cpuAlgoPixelFlow.cpp
 *
 *  Created on: 2015-02-18
 *      Author: cinnamon
 */
#include "common.h"
#include "gpuAlgoPixelFlow.h"
#include "cPFkernel.cuh"
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <stdlib.h>
#include <cstring>

// default constructor
GPU_UNOPTIMIZED::GPU_UNOPTIMIZED()
{
	// Just create an instance, init later
	m_ptr = NULL;
}

double GPU_UNOPTIMIZED::get_M0(int x, int y)
{
	//return m0[x][y];
	// use pointer to access m0 data from GPU
	if (m_ptr != NULL)
	{
		return m_ptr[y*MATRIX_DIM+x]; // z = 0, so doesn't matter right now
	}
}

void GPU_UNOPTIMIZED::setMatrixWallLoc(int x, int y, int val)
{
	cPFaddWallLocation(x,y,val);
}

void GPU_UNOPTIMIZED::cpuAlgoPixelFlow_init(double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[])
{
	cPFinit(matrixFlow, matrixWall, in_sourceLoc);
}

void GPU_UNOPTIMIZED::cpuAlgoPixelFlow_delete()
{
	cPFdelete(); // Delete some malloc'ed memory
}

void GPU_UNOPTIMIZED::cpuAlgoPixelFlow(unsigned int num_iterations)
{
	cPFcaller(num_iterations, m_ptr);
}

void GPU_UNOPTIMIZED::cpuAlgoPixelFlow_updateSource(int t)
{
	// Don't need to update it here, kernels will do it
}

void GPU_UNOPTIMIZED::cpuAlgoPixelFlow_nextStep(void)
{
	// Nothing to do here, see cPFkernel.cu instead!
}

