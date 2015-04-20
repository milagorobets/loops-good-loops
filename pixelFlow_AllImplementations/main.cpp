#include "common.h"
#include "init.h"
#include "cpuAlgoPixelFlow.h"
#include "cpuAlgoPixelFlow_v.h"
#include "cpuAlgoPixelFlow_v_1d.h"
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

// choose mode
//namespace CPU_UNOPTIMIZED {}
// CPU_UNOPTIMIZED: first implementation, uses new/delete to allocate memory
// CPU_VECTOR: version of CPU_UNOPTIMIZED with new/delete replaced by <vector>
// CPU_VECTOR_1D: version of CPU_VECTOR but <vector>s are 1D instead of 2D
using namespace TEST_SPACE;

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

int main(void)
{
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
	CPU_END;
	t1=clock();
	TEST_SPACE::cpuAlgoPixelFlow(NUM_CPU_R, matrixFlow, matrixWall, t_sourceLoc);
	TEST_SPACE::cpuAlgoPixelFlow_delete();

	CPU_END;
	printf("MATRIX_DIM %d \n", MATRIX_DIM);
	printf("%d iterations \n", TEST_SPACE::entries);

	return 0;
}





