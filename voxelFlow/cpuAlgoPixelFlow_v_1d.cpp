/*
 * cpuAlgoPixelFlow.cpp
 *
 *  Created on: 2015-02-18
 *      Author: cinnamon
 */
#include "common.h"
#include "cpuAlgoPixelFlow_v_1d.h"
//#include <array>
#include <stdio.h>
#include <math.h>
#include <vector>

namespace CPU_VECTOR_1D
{
#define PI 3.141592653589

using std::vector;

double f[4];
double source;
double sourceLoc[2];
double src_amplitude, src_frequency;
int coef = 1;

int WWAL_LENGTH = 4;
int W_LENGTH = 4;

vector<double> m0, nm0;
vector<double> m1, nm1;
vector<double> m2, nm2;
vector<double> m3, nm3;
vector<double> W, WWall;

int matrixWallLoc[MATRIX_DIM][MATRIX_DIM] = {0};

//int Index(int x, int y)
//{
//	return y*MATRIX_DIM + x;
//}

#define Index(x,y) ((y)*MATRIX_DIM + (x))
#define Index4by4(x,y) ((y)*4+(x))

void cpuAlgoPixelFlow_init(void)
{
	m0.resize(MATRIX_DIM*MATRIX_DIM);
	m1.resize(MATRIX_DIM*MATRIX_DIM);
	m2.resize(MATRIX_DIM*MATRIX_DIM);
	m3.resize(MATRIX_DIM*MATRIX_DIM);
	nm0.resize(MATRIX_DIM*MATRIX_DIM);
	nm1.resize(MATRIX_DIM*MATRIX_DIM);
	nm2.resize(MATRIX_DIM*MATRIX_DIM);
	nm3.resize(MATRIX_DIM*MATRIX_DIM);
	W.resize(16);
	WWall.resize(16);
}

void cpuAlgoPixelFlow_delete()
{
	// Nothing? Check with valgrind for memory leaks
}

void cpuAlgoPixelFlow(unsigned int num_iterations, double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[])
{
	// copy values from test matrix
	//	m0 = matrixTest[0];
	//	m1 = matrixTest[1];
	//	m2 = matrixTest[2];
	//	m3 = matrixTest[3];

	src_amplitude = 1;
	src_frequency = 1;

	sourceLoc[0] = in_sourceLoc[0]; sourceLoc[1] = in_sourceLoc[1];

	//for (int x = 0; x < 4; x++)
	//{
	//	for (int y = 0; y < 4;  y++)
	//	{
	//		W[Index(x,y)] = matrixFlow[x][y] * (coef/2.0);
	//		//printf("matrixFlow[%d][%d] = %f \n", x, y, matrixFlow[x][y]);
	//	}
	//}

	for (int x = 0; x < 4; x++)
	{
		for (int y = 0; y < 4; y++)
		{
			W[Index4by4(x,y)] = matrixFlow[x][y] * (coef/2.0);
		}
	}

	for (int x = 0; x < 4; x++)
	{
		for (int y = 0; y < 4;  y++)
		{
			WWall[Index4by4(x,y)] = matrixWall[x][y] * (coef/2.0);
		}
	}


	for (int t = 0; t < num_iterations; t++)
	{

		cpuAlgoPixelFlow_updateSource(t);
		cpuAlgoPixelFlow_nextStep();
		//printf("Iteration %d complete. \n", t);
		// display matrix values:
		//for (int i = 0; i < MATRIX_DIM; i++)
		//{
		//	//printf("Row %d: %f, %f, %f, %f, %f \n", i, m0[i][0], m0[i][1], m0[i][2], m0[i][3], m0[i][4]);
		//}
	}
}

void cpuAlgoPixelFlow_updateSource(int t)
{
	source = src_amplitude * sin(2 * PI * src_frequency * t * 0.01); //0.01 is from original java code
}

void cpuAlgoPixelFlow_nextStep(void)
{
	double f0, f1, f2, f3;
	double newF[4];
	bool isWall;
	for (int x = 0; x < MATRIX_DIM; x++)
	{
		for (int y = 0; y < MATRIX_DIM; y++)
		{
			f0 = m0[Index(x,y)];
			f1 = m1[Index(x,y)];
			f2 = m2[Index(x,y)];
			f3 = m3[Index(x,y)];

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
			isWall = matrixWallLoc[x][y]; // change this too?

			if (isWall)
			{
//				for (int i = 0; i < WWAL_LENGTH; i++)
//				{
//					newF[i] += WWall[Index(0,i)]*f0 + WWall[Index(1,i)]*f1+WWall[Index(2,i)]*f2+WWall[Index(3,i)]*f3;
//				}
				newF[0] = WWall[Index4by4(0,0)]*f0 + WWall[Index4by4(1,0)]*f1+WWall[Index4by4(2,0)]*f2+WWall[Index4by4(3,0)]*f3;
				newF[1] = WWall[Index4by4(0,1)]*f0 + WWall[Index4by4(1,1)]*f1+WWall[Index4by4(2,1)]*f2+WWall[Index4by4(3,1)]*f3;
				newF[2] = WWall[Index4by4(0,2)]*f0 + WWall[Index4by4(1,2)]*f1+WWall[Index4by4(2,2)]*f2+WWall[Index4by4(3,2)]*f3;
				newF[3] = WWall[Index4by4(0,3)]*f0 + WWall[Index4by4(1,3)]*f1+WWall[Index4by4(2,3)]*f2+WWall[Index4by4(3,3)]*f3;
			}
			else
			{
//				for (int i = 0; i < W_LENGTH; i++)
//				{
//					newF[i] += W[Index(0,i)]*f0 + W[Index(1,i)]*f1 + W[Index(2,i)]*f2 + W[Index(3,i)]*f3;
//				}
				newF[0] = W[Index4by4(0,0)]*f0 + W[Index4by4(1,0)]*f1+W[Index4by4(2,0)]*f2+W[Index4by4(3,0)]*f3;
				newF[1] = W[Index4by4(0,1)]*f0 + W[Index4by4(1,1)]*f1+W[Index4by4(2,1)]*f2+W[Index4by4(3,1)]*f3;
				newF[2] = W[Index4by4(0,2)]*f0 + W[Index4by4(1,2)]*f1+W[Index4by4(2,2)]*f2+W[Index4by4(3,2)]*f3;
				newF[3] = W[Index4by4(0,3)]*f0 + W[Index4by4(1,3)]*f1+W[Index4by4(2,3)]*f2+W[Index4by4(3,3)]*f3;
			}

			isWall = false;

			if (x < MATRIX_DIM-1) nm0[Index(x+1,y)] = newF[1];
			if (x > 0) nm1[Index(x-1,y)] = newF[0];
			if (y < MATRIX_DIM-1) nm2[Index(x,y+1)] = newF[3];
			if (y > 0) nm3[Index(x,y-1)] = newF[2];
		}
	}

	// Copy values to the matrix
	for (int x = 0; x < MATRIX_DIM; x++)
	{
		for (int y = 0; y < MATRIX_DIM; y++)
		{
			m0[Index(x,y)] = nm0[Index(x,y)];
			m1[Index(x,y)] = nm1[Index(x,y)];
			m2[Index(x,y)] = nm2[Index(x,y)];
			m3[Index(x,y)] = nm3[Index(x,y)];

			if (nm0[Index(0,y)] == 0)
			{
				m0[Index(0,y)] = nm0[Index(1,y)];
			}
			if (nm2[Index(x,0)] == 0)
			{
				m2[Index(x,0)] = nm2[Index(x,1)];
			}
			if (nm1[Index(MATRIX_DIM-1,y)] == 0)
			{
				m1[Index(MATRIX_DIM-1,y)] = nm1[Index(MATRIX_DIM-2,y)];
			}
			if (nm3[Index(x,MATRIX_DIM-1)] == 0)
			{
				m3[Index(x,MATRIX_DIM-1)] = nm3[Index(x,MATRIX_DIM-2)];
			}
		}
	}
}
} //namespace ends
