/*
 * cpuAlgoPixelFlow.cpp
 *
 *  Created on: 2015-02-18
 *      Author: cinnamon
 */
#include "common.h"
#include "cpuAlgoPixelFlow_v.h"
//#include <array>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <time.h>

namespace CPU_VECTOR
{
#define PI 3.141592653589

using std::vector;

double f[4];
double source;
double sourceLoc[2];
double src_amplitude, src_frequency;
int coef = 1;
int entries = 0;

int WWAL_LENGTH = 4;
int W_LENGTH = 4;

vector<vector<double> > m0, nm0;
vector<vector<double> > m1, nm1;
vector<vector<double> > m2, nm2;
vector<vector<double> > m3, nm3;
vector<vector<double> > W, WWall;

double W00, W01, W02, W03, W10, W11, W12, W13, W20, W21, W22, W23, W30, W31, W32, W33;

int matrixWallLoc[MATRIX_DIM][MATRIX_DIM] = {0};

void cpuAlgoPixelFlow_init(void)
{
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

void cpuAlgoPixelFlow_delete()
{
	// Nothing? Check with valgrind
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
	//		W[x][y] = matrixFlow[x][y] * (coef/2.0);
	//		//printf("matrixFlow[%d][%d] = %f \n", x, y, matrixFlow[x][y]);
	//	}
	//}

	W00 = matrixFlow[0][0]; W01 = matrixFlow[0][1]; W02 = matrixFlow[0][2]; W03 = matrixFlow[0][3];
	W10 = matrixFlow[1][0]; W11 = matrixFlow[1][1]; W12 = matrixFlow[1][2]; W13 = matrixFlow[1][3];
	W20 = matrixFlow[2][0]; W21 = matrixFlow[2][1]; W22 = matrixFlow[2][2]; W23 = matrixFlow[2][3];
	W30 = matrixFlow[3][0]; W31 = matrixFlow[3][1]; W32 = matrixFlow[3][2]; W33 = matrixFlow[3][3];

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
//		for (int i = 0; i < MATRIX_DIM; i++)
//		{
//			printf("Row %d: %f, %f, %f, %f, %f \n", i, m0[i][0], m0[i][1], m0[i][2], m0[i][3], m0[i][4]);
//		}
	}
}

void cpuAlgoPixelFlow_updateSource(int t)
{
	source = src_amplitude * sin(2 * PI * src_frequency * t * 0.01); //0.01 is from original java code
}

void cpuAlgoPixelFlow_nextStep(void)
{
	double f0, f1, f2, f3, f4;
	double newF[4] = {0};
	double newF0, newF1, newF2, newF3;
	bool isWall;
	for (int x = 1; x < MATRIX_DIM-1; x++)
	{
		for (int y = 1; y < MATRIX_DIM-1; y++)
		{
			entries++;
			f0 = m0[x][y];
			//f1 = m0[x-1][y];
			//f2 = m0[x+1][y];
			//f3 = m0[x][y-1];
			//f4 = m0[x][y+1];
			f1 = m1[x][y];
			f2 = m2[x][y];
			f3 = m3[x][y];

			/*newF[0] = 0;
			newF[1] = 0;
			newF[2] = 0;
			newF[3] = 0;*/

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
				//for (int i = 0; i < WWAL_LENGTH; i++)
				//{
				//	//newF[i] += WWall[0][i]*f0 + WWall[1][i]*f1+WWall[2][i]*f2+WWall[3][i]*f3;
				//}
				newF[0] = WWall[0][0]*f0 + WWall[1][0]*f1+WWall[2][0]*f2+WWall[3][0]*f3;
				newF[1] = WWall[0][1]*f0 + WWall[1][1]*f1+WWall[2][1]*f2+WWall[3][1]*f3;
				newF[2] = WWall[0][2]*f0 + WWall[1][2]*f1+WWall[2][0]*f2+WWall[3][2]*f3;
				newF[3] = WWall[0][3]*f0 + WWall[1][3]*f1+WWall[2][0]*f2+WWall[3][3]*f3;
			}
			else
			{
				/*for (int i = 0; i < W_LENGTH; i++)
				{
					newF[i] += W[0][i]*f0 + W[1][i]*f1 + W[2][i]*f2 + W[3][i]*f3;
				}*/
				/*newF[0] = W[0][0]*f0 + W[1][0]*f1+W[2][0]*f2+W[3][0]*f3;
				newF[1] = W[0][1]*f0 + W[1][1]*f1+W[2][1]*f2+W[3][1]*f3;
				newF[2] = W[0][2]*f0 + W[1][2]*f1+W[2][0]*f2+W[3][2]*f3;
				newF[3] = W[0][3]*f0 + W[1][3]*f1+W[2][0]*f2+W[3][3]*f3;*/
			/*	newF[0] = W[0][0]*f0 + W[0][1]*f1+W[0][2]*f2+W[0][3]*f3;
				newF[1] = W[1][0]*f0 + W[1][1]*f1+W[1][2]*f2+W[1][3]*f3;
				newF[2] = W[2][0]*f0 + W[2][1]*f1+W[2][2]*f2+W[2][3]*f3;
				newF[3] = W[3][0]*f0 + W[3][1]*f1+W[3][2]*f2+W[3][3]*f3;*/

				/*newF[0] = W00*f0 + W10*f1+W20*f2+W30*f3;
				newF[1] = W01*f0 + W11*f1+W21*f2+W31*f3;
				newF[2] = W02*f0 + W12*f1+W22*f2+W32*f3;
				newF[3] = W03*f0 + W13*f1+W23*f2+W33*f3;*/

				newF0 = W00*f0 + W10*f1+W20*f2+W30*f3;
				newF1 = W01*f0 + W11*f1+W21*f2+W31*f3;
				newF2 = W02*f0 + W12*f1+W22*f2+W32*f3;
				newF3 = W03*f0 + W13*f1+W23*f2+W33*f3;
				
				// 2.73s w/o copy
				//newF[0] = W[0][0] + W[0][1] + W[0][2] + W[0][3];
				/*newF[1] = W[1][0] + W[1][1] + W[1][2] + W[1][3];
				newF[2] = W[2][0] + W[2][1] + W[2][2] + W[2][3];
				newF[3] = W[3][0] + W[3][1] + W[3][2] + W[3][3];*/
				
				// very slow:
				/*newF[0] = W[0][0]*m0[x][y] + W[0][1]*m1[x][y]+W[0][2]*m2[x][y]+W[0][3]*m3[x][y];
				newF[1] = W[1][0]*m0[x][y] + W[1][1]*m1[x][y]+W[1][2]*m2[x][y]+W[1][3]*m3[x][y];
				newF[2] = W[2][0]*m0[x][y] + W[2][1]*m1[x][y]+W[2][2]*m2[x][y]+W[2][3]*m3[x][y];
				newF[3] = W[3][0]*m0[x][y] + W[3][1]*m1[x][y]+W[3][2]*m2[x][y]+W[3][3]*m3[x][y];*/
			}

			//isWall = false;

			/*if (x < MATRIX_DIM-1) nm0[x+1][y] = newF[1];
			if (x > 0) nm1[x-1][y] = newF[0];
			if (y < MATRIX_DIM-1) nm2[x][y+1] = newF[3];
			if (y > 0) nm3[x][y-1] = newF[2];*/
			//if (x < MATRIX_DIM-1) nm0[x][y] = newF1;
			/*if (x > 0) nm1[x-1][y] = newF0;
			if (y < MATRIX_DIM-1) nm2[x][y+1] = newF3;*/
			/*if (x > 0) int b = newF0;*/
			/*if (y < MATRIX_DIM-1) int c = newF3;
			if (y > 0) int s = newF2;*/
			nm0[x+1][y] = newF1;
			/*nm1[x-1][y] = newF[0];
			nm2[x][y+1] = newF[3];
			nm3[x][y-1] = newF[2];*/
		}
	}

	// Copy values to the matrix
	/*for (int x = 0; x < MATRIX_DIM; x++)
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
	}*/
}
} //namespace ends
