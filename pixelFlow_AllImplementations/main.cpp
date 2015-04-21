#include "common.h"
#include "init.h"
#include "cpuAlgoPixelFlow.h"
#include "cpuAlgoPixelFlow_v.h"
#include "cpuAlgoPixelFlow_v_1d.h"
#include "ref_cpuAlgoPixelFlow.h"
#include <time.h>
#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <cuda_runtime_api.h>

int position_sourceX = 3;
int position_sourceY = 3;

double frequency_source = 1;

using namespace TEST_SPACE;
//using namespace CPU_UNOPTIMIZED;
using namespace REFERENCE_SPACE;

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

#define DISPLAY_NAMESPACE printf("Running: " STRING(TEST_SPACE) ": \n")

#define STRING2(x) #x
#define STRING(x) STRING2(x)

int main(void)
{
	DISPLAY_NAMESPACE;
	
	matrixFlow_types MATRIX_FLOW_TYPE = BASIC;
	init_MatrixFlowType(&MATRIX_FLOW_TYPE, matrixFlow);

	matrixWall_types MATRIX_WALL_TYPE = TENTH;
	init_MatrixWallType(&MATRIX_WALL_TYPE, matrixWall);

	double t_sourceLoc[2];
	source_types SOURCE_TYPE = SINE; // need to update source each t
	t_sourceLoc[0] = position_sourceX;
	t_sourceLoc[1] = position_sourceY;

	// Generate some random points for the wall
	srand(time(NULL));

//	for (int i = 0; i < NUM_WALL_BLOCKS; i++)
//	{
//		int locX = rand() % MATRIX_DIM;
//		int locY = rand() % MATRIX_DIM;
//		printf("locX = %d, locY = %d. \n", locX, locY);
//		matrixWallLoc[locX][locY] = 1;
//	}


	TEST_SPACE::matrixWallLoc[0][0] = 1;
	TEST_SPACE::matrixWallLoc[0][1] = 1;
	TEST_SPACE::matrixWallLoc[0][2] = 1;
	TEST_SPACE::matrixWallLoc[0][3] = 1;
	TEST_SPACE::matrixWallLoc[0][4] = 1;

	CPU_START;

	TEST_SPACE::cpuAlgoPixelFlow_init();
	TEST_SPACE::cpuAlgoPixelFlow(NUM_CPU_R, matrixFlow, matrixWall, t_sourceLoc);
	
	CPU_END;
	printf("MATRIX_DIM %d \n", MATRIX_DIM);
	printf("%d iterations \n", TEST_SPACE::entries);

	// If user wants to check the output against working code
#if CHECK_OUTPUT
	printf("\n Beginning Reference Computations... \n");
	REFERENCE_SPACE::matrixWallLoc[0][0] = 1;
	REFERENCE_SPACE::matrixWallLoc[0][1] = 1;
	REFERENCE_SPACE::matrixWallLoc[0][2] = 1;
	REFERENCE_SPACE::matrixWallLoc[0][3] = 1;
	REFERENCE_SPACE::matrixWallLoc[0][4] = 1;

	REFERENCE_SPACE::cpuAlgoPixelFlow_init();
	REFERENCE_SPACE::cpuAlgoPixelFlow(NUM_CPU_R, matrixFlow, matrixWall, t_sourceLoc);
	
	printf("\n Comparing outputs... \n");
	int wrong = 0;
	for (int x = 0; x < MATRIX_DIM; x++)
	{
		for (int y = 0; y < MATRIX_DIM; y++)
		{
			double refM0 = REFERENCE_SPACE::get_M0(x,y);
			double tM0 = TEST_SPACE::get_M0(x,y);
			printf("ref: %f \n", refM0);
			//printf("ref: %f, calc: %f \n", refM0, tM0);
			//if (REFERENCE_SPACE::get_M0(x,y) != TEST_SPACE::get_M0(x,y)) // need tolerance
			//{
			//	wrong++;
			//}
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
	REFERENCE_SPACE::cpuAlgoPixelFlow_delete();

#endif

	TEST_SPACE::cpuAlgoPixelFlow_delete();

	return 0;
}





