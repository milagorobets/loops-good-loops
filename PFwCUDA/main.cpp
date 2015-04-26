#include "common.h"
#include "init.h"
#include "cpuAlgoPixelFlow.h"
#include "cpuAlgoPixelFlow_v.h"
#include "cpuAlgoPixelFlow_v_1d.h"
#include "ref_cpuAlgoPixelFlow.h"
#include "gpuAlgoPixelFlow.h"
//#include "cPFkernel.cuh"
#include <time.h>
#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>

int position_sourceX = 3;
int position_sourceY = 3;

double frequency_source = 1;


double matrixFlow[4][4] = {{1.0,-1.0,1.0,1.0},
        {-1.0,1.0,1.0,1.0},
        {1.0,1.0,1.0,-1.0},
        {1.0,1.0,-1.0,1.0}};

double matrixWall[4][4] = {{1.0,-1.0,1.0,1.0},
        {-1.0,1.0,1.0,1.0},
        {1.0,1.0,1.0,-1.0},
        {1.0,1.0,-1.0,1.0}};

#define CPU_START clock_t t1; t1=clock();
#define CPU_END {long int final=clock()-t1; printf("CPU took %li ticks (%f seconds) \n", final, ((float)final)/CLOCKS_PER_SEC);}

#define STRING2(x) #x
#define STRING(x) STRING2(x)

int main(void)
{
	// Display the name of the class we are testing
	printf("This project uses CUDA. April 23, 2015. \n");
	printf("Testing: " STRING(TEST_CLASS) ": \n");

	printf("size of double: %d, size of float: %d \n", sizeof(double), sizeof(float));

	matrixFlow_types MATRIX_FLOW_TYPE = BASIC;
	init_MatrixFlowType(&MATRIX_FLOW_TYPE, matrixFlow);

	matrixWall_types MATRIX_WALL_TYPE = TENTH;
	init_MatrixWallType(&MATRIX_WALL_TYPE, matrixWall);

	double t_sourceLoc[2];
	source_types SOURCE_TYPE = SINE; 
	t_sourceLoc[0] = position_sourceX;
	t_sourceLoc[1] = position_sourceY;

	// Generate some random points for the wall
	//srand(time(NULL));

//	for (int i = 0; i < NUM_WALL_BLOCKS; i++)
//	{
//		int locX = rand() % MATRIX_DIM;
//		int locY = rand() % MATRIX_DIM;
//		printf("locX = %d, locY = %d. \n", locX, locY);
//		matrixWallLoc[locX][locY] = 1;
//	}

	TEST_CLASS cpu_test;
	cpu_test.cpuAlgoPixelFlow_init(matrixFlow, matrixWall, t_sourceLoc);

	cpu_test.setMatrixWallLoc(0,0,1);
	cpu_test.setMatrixWallLoc(0,1,1);
	cpu_test.setMatrixWallLoc(0,2,1);
	cpu_test.setMatrixWallLoc(0,3,1);
	cpu_test.setMatrixWallLoc(0,4,1);

	CPU_START;
	cpu_test.cpuAlgoPixelFlow(NUM_CPU_R);
	CPU_END;

	// If user wants to check the output against working code
#if CHECK_OUTPUT
	printf("\nChecking with: " STRING(REFERENCE_CLASS) " \n");
	printf("\nBeginning Reference Computations... \n");
	init_MatrixFlowType(&MATRIX_FLOW_TYPE, matrixFlow);
	init_MatrixWallType(&MATRIX_WALL_TYPE, matrixWall);
	SOURCE_TYPE = SINE; // need to update source each t
	t_sourceLoc[0] = position_sourceX;
	t_sourceLoc[1] = position_sourceY;
	REFERENCE_CLASS cpu_ref;
	cpu_ref.cpuAlgoPixelFlow_init(matrixFlow, matrixWall, t_sourceLoc);
	cpu_ref.setMatrixWallLoc(0,0,1);
	cpu_ref.setMatrixWallLoc(0,1,1);
	cpu_ref.setMatrixWallLoc(0,2,1);
	cpu_ref.setMatrixWallLoc(0,3,1);
	cpu_ref.setMatrixWallLoc(0,4,1);

	cpu_ref.cpuAlgoPixelFlow(NUM_CPU_R);
	
	printf("\nComparing outputs... (TOLERANCE: %f) \n", CHECK_TOLERANCE);
	int wrong = 0;
	for (int x = 0; x < MATRIX_DIM; x++)
	{
		for (int y = 0; y < MATRIX_DIM; y++)
		{
			double refM0 = cpu_ref.get_M0(x,y);
			double tM0 = cpu_test.get_M0(x,y);
			if (!((tM0 < (refM0 + CHECK_TOLERANCE)) && (tM0 > (refM0 - CHECK_TOLERANCE)))) // need tolerance
			{
				printf("Expected %f, but was %f \n", refM0, tM0);
				wrong++;
			}
		}
	}
	if (wrong > 0) 
	{
		printf("FAILURE! %d entries did not match.\n", wrong);
	}
	else
	{
		printf("SUCCESS! All entries matched. \n");
	}
	cpu_ref.cpuAlgoPixelFlow_delete();

#endif

	cpu_test.cpuAlgoPixelFlow_delete();


	return 0;
}





