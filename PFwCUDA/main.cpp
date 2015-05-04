#include "common.h"
#include "init.h"
#include "cpuAlgoPixelFlow.h"
#include "cpuAlgoPixelFlow_v.h"
#include "cpuAlgoPixelFlow_v_1d.h"

#include "ref_cpuAlgoPixelFlow.h"
//#if (TEST_CLASS == GPU_UNOPTIMIZED)
//#include "gpuAlgoPixelFlow.h"
//#elif (TEST_CLASS == GPU_PTR)
#include "gpuPF_ptr.h"
//#include "cPFkernel.cuh"
//#endif
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

float frequency_source = 1;


float matrixFlow[4][4] = {{1.0,-1.0,1.0,1.0},
        {-1.0,1.0,1.0,1.0},
        {1.0,1.0,1.0,-1.0},
        {1.0,1.0,-1.0,1.0}};

float matrixWall[4][4] = {{1.0,-1.0,1.0,1.0},
        {-1.0,1.0,1.0,1.0},
        {1.0,1.0,1.0,-1.0},
        {1.0,1.0,-1.0,1.0}};

#define CPU_START clock_t t1; t1=clock();
#define CPU_END {long int final=clock()-t1; printf("Environment Under Test took %f seconds \n", ((float)final)/CLOCKS_PER_SEC);}

#define STRING2(x) #x
#define STRING(x) STRING2(x)

int main(void)
{
	// Display the name of the class we are testing
	printf("This project uses CUDA. April 28, 2015. \n");
	printf("Testing: " STRING(TEST_CLASS) ": \n");

	matrixFlow_types MATRIX_FLOW_TYPE = BASIC;
	init_MatrixFlowType(&MATRIX_FLOW_TYPE, matrixFlow);

	matrixWall_types MATRIX_WALL_TYPE = TENTH;
	init_MatrixWallType(&MATRIX_WALL_TYPE, matrixWall);

	float t_sourceLoc[2];
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

	long int t2 = clock();
	cpu_ref.cpuAlgoPixelFlow(NUM_CPU_R);
	long int finalt2 = clock()- t2; printf("Reference calculations took %f seconds \n", ((float)finalt2)/CLOCKS_PER_SEC);
	printf("\nComparing outputs... (TOLERANCE: %f) \n", CHECK_TOLERANCE);
	int wrong = 0;
	double refM0, tM0;
	for (int x = 0; x < MATRIX_DIM; x++)
	{
		for (int y = 0; y < MATRIX_DIM; y++)
		{
			refM0 = cpu_ref.get_M0(x,y);
			tM0 = cpu_test.get_M0(x,y);
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

	cudaDeviceReset();


	return 0;
}





