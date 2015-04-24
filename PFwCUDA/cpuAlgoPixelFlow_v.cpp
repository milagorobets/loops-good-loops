/*
 * cpuAlgoPixelFlow.cpp
 *
 *  Created on: 2015-02-18
 *      Author: cinnamon
 */
#include "common.h"
#include <vector>
#include "cpuAlgoPixelFlow_v.h"
//#include <array>
#include <stdio.h>
#include <math.h>
#include <time.h>


CPU_VECTOR::CPU_VECTOR()
{
	cpuAlgoPixelFlow_init();
}

double CPU_VECTOR::get_M0(int x, int y)
{
	return m0[x][y];
}

void CPU_VECTOR::setMatrixWallLoc(int x, int y, int val)
{
	#if (WALL_MEMORY==MEM_STACK)
	matrixWallLoc[x][y] = val;
	#elif (WALL_MEMORY==MEM_MAP)
	mapWallLoc[x][y] = val;
	#elif(WALL_MEMORY==MEM_HEAP)
	heapWallLoc[x][y] = val;
	#endif
}

void CPU_VECTOR::cpuAlgoPixelFlow_init(void)
{
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

	m0.resize(MATRIX_DIM);
	m1.resize(MATRIX_DIM);
	m2.resize(MATRIX_DIM);
	m3.resize(MATRIX_DIM);
	nm0.resize(MATRIX_DIM);
	nm1.resize(MATRIX_DIM);
	nm2.resize(MATRIX_DIM);
	nm3.resize(MATRIX_DIM);
	W.resize(4);
	WWall.resize(4);

	for (int i = 0; i < MATRIX_DIM; ++i)
	{
		m0[i].resize(MATRIX_DIM);
		m1[i].resize(MATRIX_DIM);
		m2[i].resize(MATRIX_DIM);
		m3[i].resize(MATRIX_DIM);
		nm0[i].resize(MATRIX_DIM);
		nm1[i].resize(MATRIX_DIM);
		nm2[i].resize(MATRIX_DIM);
		nm3[i].resize(MATRIX_DIM);
	}

	for (int i = 0; i < 4; ++i)
	{
		W[i].resize(4);
		WWall[i].resize(4);
	}
}

void CPU_VECTOR::cpuAlgoPixelFlow_delete()
{
	// Nothing? Check with valgrind
}

void CPU_VECTOR::cpuAlgoPixelFlow(unsigned int num_iterations, double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[])
{
		// copy values from test matrix
	//	m0 = matrixTest[0];
	//	m1 = matrixTest[1];
	//	m2 = matrixTest[2];
	//	m3 = matrixTest[3];

	src_amplitude = 1.0;
	src_frequency = 1.0;

	sourceLoc[0] = in_sourceLoc[0]; sourceLoc[1] = in_sourceLoc[1];

	for (int x = 0; x < 4; x++)
	{
		for (int y = 0; y < 4;  y++)
		{
			W[x][y] = matrixFlow[x][y] * (coef/2.0);
			//printf("matrixFlow[%d][%d] = %f, W = %f\n", x, y, matrixFlow[x][y], W[x][y]);
		}
	}

	for (int x = 0; x < 4; x++)
	{
		for (int y = 0; y < 4;  y++)
		{
			WWall[x][y] = matrixWall[x][y] * (coef/2.0);
		}
	}

	source = 0;
	for (int t = 0; t < num_iterations; t++)
	{
		/*for (int i = 0; i < MATRIX_DIM; i++)
		{
			printf("Row %d: %.9G, %.9G, %.9G, %.9G, %.9G \n", i, m0[i][0], m0[i][1], m0[i][2], m0[i][3], m0[i][4]);
		}*/
		cpuAlgoPixelFlow_updateSource(t);
		cpuAlgoPixelFlow_nextStep();
//		printf("Iteration %d complete. \n", t);
//		// display matrix values:		
	}
}

void CPU_VECTOR::cpuAlgoPixelFlow_updateSource(int t)
{
	source = src_amplitude * sin(2 * PI * src_frequency * (double)(t) * 0.01); //0.01 is from original java code
}

void CPU_VECTOR::cpuAlgoPixelFlow_nextStep(void)
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
