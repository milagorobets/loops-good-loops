/*
 * cpuAlgoPixelFlow.cpp
 *
 *  Created on: 2015-02-18
 *      Author: cinnamon
 */
#include "common.h"
#include "ref_cpuAlgoPixelFlow.h"
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <stdlib.h>
#include <cstring>

// default constructor
CPU_REFERENCE::CPU_REFERENCE()
{
	cpuAlgoPixelFlow_init();
}

double CPU_REFERENCE::get_M0(int x, int y)
{
	return m0[x][y];
}

void CPU_REFERENCE::setMatrixWallLoc(int x, int y, int val)
{
	matrixWallLoc[x][y] = val;
}

void CPU_REFERENCE::cpuAlgoPixelFlow_init(void)
{
	// Initialize some values
	coef = 1;
	entries = 0;
	WWAL_LENGTH = 4;
	W_LENGTH = 4;

	for (int x = 0; x < MATRIX_DIM; x++)
	{
		for (int y = 0; y < MATRIX_DIM; y++)
		{
			matrixWallLoc[x][y] = 0;
		}
	}

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
}

void CPU_REFERENCE::cpuAlgoPixelFlow_delete()
{
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

void CPU_REFERENCE::cpuAlgoPixelFlow(unsigned int num_iterations, double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[])
{
	// copy values from test matrix
	//	m0 = matrixTest[0];
	//	m1 = matrixTest[1];
	//	m2 = matrixTest[2];
	//	m3 = matrixTest[3];

	src_amplitude = 1;
	src_frequency = 1;

	sourceLoc[0] = in_sourceLoc[0]; sourceLoc[1] = in_sourceLoc[1];

	for (int x = 0; x < 4; x++)
	{
		for (int y = 0; y < 4;  y++)
		{
			W[x][y] = matrixFlow[x][y] * (coef/2.0);
			//printf("matrixFlow[%d][%d] = %f \n", x, y, matrixFlow[x][y]);
		}
	}

	for (int x = 0; x < 4; x++)
	{
		for (int y = 0; y < 4;  y++)
		{
			WWall[x][y] = matrixWall[x][y] * (coef/2.0);
		}
	}


	for (int t = 0; t < num_iterations; t++)
	{

		cpuAlgoPixelFlow_updateSource(t);
		cpuAlgoPixelFlow_nextStep();
//		printf("Iteration %d complete. \n", t);
//		// display matrix values:
		/*for (int i = 0; i < MATRIX_DIM; i++)
		{
			printf("Row %d: %f, %f, %f, %f, %f \n", i, m0[i][0], m0[i][1], m0[i][2], m0[i][3], m0[i][4]);
		}*/
	}
}

void CPU_REFERENCE::cpuAlgoPixelFlow_updateSource(int t)
{
	source = src_amplitude * sin(2 * PI * src_frequency * t * 0.01); //0.01 is from original java code
}

void CPU_REFERENCE::cpuAlgoPixelFlow_nextStep(void)
{
	double f0, f1, f2, f3;
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
			isWall = matrixWallLoc[x][y];

			if (isWall)
			{
				for (int i = 0; i < WWAL_LENGTH; i++)
				{
					newF[i] += WWall[0][i]*f0 + WWall[1][i]*f1+WWall[2][i]*f2+WWall[3][i]*f3;
				}
			}
			else
			{
				for (int i = 0; i < W_LENGTH; i++)
				{
					newF[i] += W[0][i]*f0 + W[1][i]*f1 + W[2][i]*f2 + W[3][i]*f3;
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

