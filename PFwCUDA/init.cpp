#include "init.h"

void init_MatrixFlowType(matrixFlow_types* MATRIX_TYPE, float matrix[][4])
{
	if (*MATRIX_TYPE == BASIC)
	{
		float matrixFlow[4][4] = {{1.0,-1.0,1.0,1.0},
					{-1.0,1.0,1.0,1.0},
					{1.0,1.0,1.0,-1.0},
					{1.0,1.0,-1.0,1.0}};
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				matrix[i][j]=matrixFlow[i][j];
			}
		}
		
	}
	else if (*MATRIX_TYPE == POSITIVE)
	{
		float matrixFlow[4][4] = {{1.0,1.0,1.0,1.0},
		        {1.0,1.0,1.0,1.0},
		        {1.0,1.0,1.0,1.0},
		        {1.0,1.0,1.0,1.0}};
		for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						matrix[i][j]=matrixFlow[i][j];
					}
				}
	}
	else if (*MATRIX_TYPE == NEGATIVE)
	{
		float matrixFlow[4][4] = {{-1.0,-1.0,-1.0,-1.0},
		        {-1.0,-1.0,-1.0,-1.0},
		        {-1.0,-1.0,-1.0,-1.0},
		        {-1.0,-1.0,-1.0,-1.0}};
		for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						matrix[i][j]=matrixFlow[i][j];
					}
				}
	}
	else if (*MATRIX_TYPE == RANDOM)
	{
		float matrixFlow[4][4] = {{1.0,2.0,3.0,4.0},
		        {5.0,6.0,7.0,8.0},
		        {9.0,10.0,11.0,12.0},
		        {13.0,14.0,15.0,16.0}};
		for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						matrix[i][j]=matrixFlow[i][j];
					}
				}
	}
	else
	{
		float matrixFlow[4][4] = {{1.0,2.0,3.0,4.0},
					{5.0,6.0,7.0,8.0},
					{9.0,10.0,11.0,12.0},
					{13.0,14.0,15.0,16.0}};
		for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						matrix[i][j]=matrixFlow[i][j];
					}
				}
	}
}

void init_MatrixWallType(matrixWall_types* MATRIX_TYPE, float matrix[][4])
{
	if (*MATRIX_TYPE == TENTH)
	{
		float matrixWall[4][4] = {{0.1,-0.1,0.1,0.1},
		    	{-0.1,0.1,0.1,0.1},
		    	{0.1,0.1,0.1,-0.1},
		    	{0.1,0.1,-0.1,0.1}};
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				matrix[i][j]=matrixWall[i][j];
			}
		}
	}
	else if (*MATRIX_TYPE == ZERO)
	{
		float matrixWall[4][4] = {{0.0,-1.0,0.0,0.0},
		        {-1.0,0.0,0.0,0.0},
		        {0.0,0.0,0.0,-1.0},
		        {0.0,0.0,-1.0,0.0}};
		for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						matrix[i][j]=matrixWall[i][j];
					}
				}
	}
	else if (*MATRIX_TYPE == MIX)
	{
		float matrixWall[4][4] = {{0.7,-0.2,1.4,0.4},
		        {-1.3,0.2,0.6,4.0},
		        {1.2,0.1,2.3,-0.4},
		        {0.6,1.4,-1.8,0.3}};
		for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						matrix[i][j]=matrixWall[i][j];
					}
				}
	}
	else
	{
		float matrixWall[4][4] = {{0.7,-0.2,1.4,0.4},
		        {-1.3,0.2,0.6,4.0},
		        {1.2,0.1,2.3,-0.4},
		        {0.6,1.4,-1.8,0.3}};
		for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						matrix[i][j]=matrixWall[i][j];
					}
				}
	}


}

