/*
 * cpuAlgoPixelFlow.cpp
 *
 *  Created on: 2015-02-18
 *      Author: cinnamon
 */
#include "common.h"
#include "cpuAlgoPixelFlow.h"
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <stdlib.h>
#include <cstring>

// default constructor
CPU_UNOPTIMIZED::CPU_UNOPTIMIZED()
{
	/*cpuAlgoPixelFlow_init();*/
}

double CPU_UNOPTIMIZED::get_M0(int x, int y)
{
	return m0[x][y];
}

void CPU_UNOPTIMIZED::setMatrixWallLoc(int x, int y, int val)
{
	#if (WALL_MEMORY==MEM_STACK)
	matrixWallLoc[x][y] = val;
	#elif (WALL_MEMORY==MEM_MAP)
	mapWallLoc[x][y] = val;
	#elif(WALL_MEMORY==MEM_HEAP)
	heapWallLoc[x][y] = val;
	#endif
}

void CPU_UNOPTIMIZED::cpuAlgoPixelFlow_init(double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[])
{
	// Initialize some values
	coef = 1;
	entries = 0;
	WWAL_LENGTH = 4;
	W_LENGTH = 4;

	#if (WALL_MEMORY==MEM_STACK)
	for (int x = 0; x < MATRIX_DIM; x++)
	{
		for (int y = 0; y < MATRIX_DIM; y++)
		{
			matrixWallLoc[x][y] = 0;
		}
	}
	#elif (WALL_MEMORY==MEM_HEAP)
	heapWallLoc = new bool * [MATRIX_DIM];
	for (int x = 0; x < MATRIX_DIM; x++)
	{
		heapWallLoc[x] = new bool [MATRIX_DIM];
		memset(heapWallLoc[x], 0, MATRIX_DIM*(sizeof *heapWallLoc[x]));
	}
	#endif

	m0 = new double * [MATRIX_DIM];
	m1 = new double * [MATRIX_DIM];
	m2 = new double * [MATRIX_DIM];
	m3 = new double * [MATRIX_DIM];

	nm0 = new double * [MATRIX_DIM];
	nm1 = new double * [MATRIX_DIM];
	nm2 = new double * [MATRIX_DIM];
	nm3 = new double * [MATRIX_DIM];

	W = new double * [4];
	WWall = new double * [4];

	for (int i = 0; i < MATRIX_DIM; ++i)
	{
		m0[i] = new double [MATRIX_DIM];
		memset(m0[i], 0, MATRIX_DIM*(sizeof *m0[i]));
		m1[i] = new double [MATRIX_DIM];
		memset(m1[i], 0, MATRIX_DIM*(sizeof *m1[i]));
		m2[i] = new double [MATRIX_DIM];
		memset(m2[i], 0, MATRIX_DIM*(sizeof *m2[i]));
		m3[i] = new double [MATRIX_DIM];
		memset(m3[i], 0, MATRIX_DIM*(sizeof *m3[i]));
		nm0[i] = new double [MATRIX_DIM];
		memset(nm0[i], 0, MATRIX_DIM*(sizeof *nm0[i]));
		nm1[i] = new double [MATRIX_DIM];
		memset(nm1[i], 0, MATRIX_DIM*(sizeof *nm1[i]));
		nm2[i] = new double [MATRIX_DIM];
		memset(nm2[i], 0, MATRIX_DIM*(sizeof *nm2[i]));
		nm3[i] = new double [MATRIX_DIM];
		memset(nm3[i], 0, MATRIX_DIM*(sizeof *nm3[i]));
	}
	for (int i = 0; i < 4; ++i)
	{
		W[i] = new double [4];
		WWall[i] = new double [4];
		memset(W[i], 0, 4*(sizeof *W[i]));
		memset(WWall[i], 0, 4*(sizeof *WWall[i]));
	}

	src_amplitude = 1.0;
	src_frequency = 1.0;

	sourceLoc[0] = in_sourceLoc[0]; sourceLoc[1] = in_sourceLoc[1];

	for (int x = 0; x < 4; x++)
	{
		for (int y = 0; y < 4;  y++)
		{
			W[x][y] = matrixFlow[x][y] * (coef/2.0);
		}
	}

	for (int x = 0; x < 4; x++)
	{
		for (int y = 0; y < 4;  y++)
		{
			WWall[x][y] = matrixWall[x][y] * (coef/2.0);
		}
	}
}

void CPU_UNOPTIMIZED::cpuAlgoPixelFlow_delete()
{
#if (WALL_MEMORY == MEM_HEAP)
	for (int x = 0; x < MATRIX_DIM; ++x)
	{
		delete [] heapWallLoc[x];
	}
	delete [] heapWallLoc;
#endif
	for (int i = 0; i < MATRIX_DIM; ++i)
	{
		delete [] m0[i];
		delete [] m1[i];
		delete [] m2[i];
		delete [] m3[i];
		delete [] nm0[i];
		delete [] nm1[i];
		delete [] nm2[i];
		delete [] nm3[i];
	}
	for (int i = 0; i < 4; ++i)
	{
		delete [] W[i];
		delete [] WWall[i];
	}

	delete [] m0;
	delete [] m1;
	delete [] m2;
	delete [] m3;
	delete [] nm0;
	delete [] nm1;
	delete [] nm2;
	delete [] nm3;
	delete [] W;
	delete [] WWall;
}

void CPU_UNOPTIMIZED::cpuAlgoPixelFlow(unsigned int num_iterations)
{
	source = 0;
	for (int t = 0; t < num_iterations; t++)
	{
		cpuAlgoPixelFlow_updateSource(t);
		cpuAlgoPixelFlow_nextStep();	
		//for (int i = 0; i < MATRIX_DIM; i++)
		//{
		//	printf("Source: %f, Row %d: ", source, i);
		//	for (int j = 0; j < MATRIX_DIM; j++)
		//	{
		//		printf("%f ", m0[j][i]);
		//	}
		//	printf("\n");
		//}
	}
}

void CPU_UNOPTIMIZED::cpuAlgoPixelFlow_updateSource(int t)
{
	source = src_amplitude * sin(2 * PI * src_frequency * (double)(t) * 0.01); //0.01 is from original java code
}

void CPU_UNOPTIMIZED::cpuAlgoPixelFlow_nextStep(void)
{
	double f0 = 0, f1 = 0, f2 = 0, f3 = 0;
	double newF[4];
	bool isWall;
	for (int x = 0; x < MATRIX_DIM; x++)
	{
		for (int y = 0; y < MATRIX_DIM; y++)
		{
			entries++;
			f0 = m0[x][y];
			f1 = m1[x][y];
			f2 = m2[x][y];
			f3 = m3[x][y];
			
			newF[0] = 0;
			newF[1] = 0;
			newF[2] = 0;
			newF[3] = 0;

			// check if source
			if (x == sourceLoc[0] && y == sourceLoc[1])
			{
				f0 = source;
				f1 = source;
				f2 = source;
				f3 = source;
			}

			// check if pixel is a wall
			#if (WALL_MEMORY==MEM_STACK)
			isWall = (bool)(matrixWallLoc[x][y]);
			#elif (WALL_MEMORY == MEM_MAP)
			isWall = (mapWallLoc.find(x) != mapWallLoc.end() && mapWallLoc[x].find(y) != mapWallLoc[x].end());
			#elif (WALL_MEMORY == MEM_HEAP)
			isWall = (bool)(heapWallLoc[x][y]);
			#endif

			if (isWall)
			{
				for (int i = 0; i < WWAL_LENGTH; i++)
				{
					newF[i] = WWall[0][i]*f0 + WWall[1][i]*f1+WWall[2][i]*f2+WWall[3][i]*f3;
				}
			}
			else
			{
				for (int i = 0; i < W_LENGTH; i++)
				{
					/*double w0i = W[0][i];
					double w1i = W[1][i];
					double w2i = W[2][i];
					double w3i = W[3][i];
					double testf = f0*w0i;*/
					newF[i] = W[0][i]*f0 + W[1][i]*f1 + W[2][i]*f2 + W[3][i]*f3;
					//newF[i] = w0i*f0 + w1i*f1 + w2i*f2 + w3i*f3;
				}
			}

			isWall = false;

			if (x < MATRIX_DIM-1) nm0[x+1][y] = newF[1];
			if (x > 0) nm1[x-1][y] = newF[0];
			if (y < MATRIX_DIM-1) nm2[x][y+1] = newF[3];
			if (y > 0) nm3[x][y-1] = newF[2];
		}
	}

	// Copy values to the matrix
	for (int x = 0; x < MATRIX_DIM; x++)
	{
		for (int y = 0; y < MATRIX_DIM; y++)
		{
			m0[x][y] = nm0[x][y];
			m1[x][y] = nm1[x][y];
			m2[x][y] = nm2[x][y];
			m3[x][y] = nm3[x][y];

			if (nm0[0][y] == 0)
			{
				m0[0][y] = nm0[1][y];
			}
			if (nm2[x][0] == 0)
			{
				m2[x][0] = nm2[x][1];
			}
			if (nm1[MATRIX_DIM-1][y] == 0)
			{
				m1[MATRIX_DIM-1][y] = nm1[MATRIX_DIM-2][y];
			}
			if (nm3[x][MATRIX_DIM-1] == 0)
			{
				m3[x][MATRIX_DIM-1] = nm3[x][MATRIX_DIM-2];
			}
		}
	}
}

