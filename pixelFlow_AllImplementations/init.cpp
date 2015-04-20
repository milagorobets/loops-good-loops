#include "init.h"

void init_MatrixFlowType(matrixFlow_types* MATRIX_TYPE, double matrix[][4])
{
	if (*MATRIX_TYPE == BASIC)
	{
		double matrixFlow[4][4] = {{1.0,-1.0,1.0,1.0},
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
		double matrixFlow[4][4] = {{1.0,1.0,1.0,1.0},
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
		double matrixFlow[4][4] = {{-1.0,-1.0,-1.0,-1.0},
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
		double matrixFlow[4][4] = {{1.0,2.0,3.0,4.0},
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
		double matrixFlow[4][4] = {{1.0,2.0,3.0,4.0},
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

void init_MatrixWallType(matrixWall_types* MATRIX_TYPE, double matrix[][4])
{
	if (*MATRIX_TYPE == TENTH)
	{
		double matrixWall[4][4] = {{0.1,-0.1,0.1,0.1},
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
		double matrixWall[4][4] = {{0.0,-1.0,0.0,0.0},
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
		double matrixWall[4][4] = {{0.7,-0.2,1.4,0.4},
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
		double matrixWall[4][4] = {{0.7,-0.2,1.4,0.4},
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

