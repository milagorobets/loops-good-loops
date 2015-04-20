#include <time.h>
#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <cstring>
#include <omp.h>

#define DIM 512
#define ITERATIONS 1000

#define CPU_START clock_t t1; t1=clock();
#define CPU_END {long int final=clock()-t1; printf("CPU took %li ticks (%f seconds) \n", final, ((float)final)/CLOCKS_PER_SEC);}

#define START_TIMING_VECTOR clock_t t1; t1=clock();
#define STOP_TIMING_VECTOR {long int final_v=clock()-t1; printf("VECTOR took %li ticks (%f seconds) \n", final_v, ((float)final_v)/CLOCKS_PER_SEC);}

#define START_TIMING_ND clock_t t2; t2=clock();
#define STOP_TIMING_ND {long int final_nd=clock()-t2; printf("NEW/DELETE took %li ticks (%f seconds) \n", final_nd, ((float)final_nd)/CLOCKS_PER_SEC);}

#define START_TIMING_MALLOC clock_t t3; t3=clock();
#define STOP_TIMING_MALLOC {long int final_m=clock()-t3; printf("MALLOC took %li ticks (%f seconds) \n", final_m, ((float)final_m)/CLOCKS_PER_SEC);}

// vectors
using std::vector;
vector<vector<double> > v_matrix;
vector<vector<double> > v_res_matrix;
vector<vector<double> > v_coefficients;
//double ** v_res_matrix;


//double matrixFlow[4][4] = {{1.0,-1.0,1.0,1.0},
//        {-1.0,1.0,1.0,1.0},
//        {1.0,1.0,1.0,-1.0},
//        {1.0,1.0,-1.0,1.0}};

int main(void)
{
	// new/delete
	double ** d_matrix, ** d_res_matrix;
	double ** m_matrix = (double **) malloc(DIM * sizeof(double *));
	double ** m_res_matrix = (double **) malloc(DIM*sizeof(double*));
	for (int i = 0; i < DIM; i++)
	{
		m_matrix[i] = (double*)malloc (DIM*sizeof(double));
		m_res_matrix[i] = (double*) malloc (DIM*sizeof(double));
	}

	v_matrix.resize(DIM);
	for (int i = 0; i < DIM; i++)
	{
		v_matrix[i].resize(DIM);
	}
	v_res_matrix.resize(DIM);
	for (int i = 0; i < DIM; i++)
	{
		v_res_matrix[i].resize(DIM);
	}

	for (int x = 0; x < DIM; x++)
	{
		for (int y = 0; y < DIM; y++)
		{
			v_res_matrix[x][y] = 0.0;
			v_matrix[x][y] = 0.0;
		}
	}

	d_res_matrix = new double * [DIM];
	d_matrix = new double * [DIM];
	for (int i = 0; i < DIM; i++)
	{
		d_matrix[i] = new double [DIM];
		memset(d_matrix[i], 0, DIM*(sizeof *d_matrix[i]));
		d_res_matrix[i] = new double[DIM];
		memset(d_res_matrix[i], 0, DIM*(sizeof *d_res_matrix[i]));
	}
	d_matrix[20][45] = 1; // start somewhere

	// vector calculations
	double temp1, temp2, temp3;
	double f0, f1, f2, f3, f4;
	double * save_write_loc;
	v_matrix[20][45] = 1; // start somewhere
	//START_TIMING_VECTOR;
	//for (int iter = 0; iter < ITERATIONS; iter++)
	//{
	//	for (int x = 1; x < DIM-1; x++) // avoid boundary cases for this example
	//	{
	//		for (int y = 1; y < DIM-1; y++)
	//		{
	//			f1 = v_matrix[x-1][y];
	//			f0 = v_matrix[x][y];	
	//			save_write_loc = &v_res_matrix[x][y];
	//			//temp1 = f0*0.6 + f1*0.1;
	//			f2 = v_matrix[x+1][y];
	//			//f3 = 0; f4 = 0;
	//			f3 = v_matrix[x][y-1];
	//			//temp1 = f0*0.6 + f1*0.1;
	//			//temp2 = f2*0.1 + f3*0.1;
	//			f4 = v_matrix[x][y+1];
	//			//temp3 = f4*0.1;

	//			
	//			//temp2 = f2*0.1 + f3*0.1;
	//			//temp3 = f4*0.1;
	//			//v_res_matrix[x][y] = f0*0.6 + f1*0.1 + f2*0.1 + f3*0.1 + f4*0.1;
	//			*save_write_loc = f0*0.6 + f1*0.1 + f2*0.1 + f3*0.1 + f4*0.1;
	//			/*v_res_matrix[x][y] = f0*0.6 + f1*0.1 + f2*0.1 + f3*0.1 + f4*0.1;
	//			v_res_matrix[x][y] = f0*0.6 + f1*0.1 + f2*0.1 + f3*0.1 + f4*0.1;*/
	//			//v_res_matrix[x][y] = temp1 + temp2 + temp3;
	//		}
	//	}
	//	for (int x = 1; x < DIM-1; x++)
	//	{
	//		for (int y = 1; y < DIM-1; y++)
	//		{
	//			v_matrix[x][y] = v_res_matrix[x][y];
	//		}
	//	}
	//}
	//STOP_TIMING_VECTOR;

	double a;
	double * pf0, *pf1, *pf2, *pf3, *pf4;
	double * s_m, * s_r_m;
	START_TIMING_ND;
	int time_test1 = clock();

	for (int iter = 0; iter < ITERATIONS; iter++)
	{
		#pragma omp parallel 
		{
		#pragma omp for schedule(static) nowait
		for (int x = 1; x < DIM-1; x++) // avoid boundary cases for this example
		{
			//f3 = d_matrix[x][0];
			pf3 = &d_matrix[x][0];
			pf0 = &d_matrix[x][1];
			pf1 = &d_matrix[x-1][1];
			pf2 = &d_matrix[x+1][1];
			pf4 = &d_matrix[x][2];
			save_write_loc = &d_res_matrix[x][1];
			for (int y = 1; y < DIM-1; y++)
			{

				//double f0 = d_matrix[x][y];				
				//double f1 = d_matrix[x-1][y];
				
				//double f2 = d_matrix[x+1][y];
				//double f3 = d_matrix[x+1][y-1];
				//double f4 = d_matrix[x][y+1];
				
				////pf0++;
				
				////pf1++;
				////////pf0++;
				
				////pf2++;
				////////f0 += f1;
				////////pf1++;
				
				////////pf2++;
				////pf3++;
				
				////pf4++;

				//d_res_matrix[x][y] = f0*0.6 + f1*0.1 + f2*0.1 + f3*0.1 + f4*0.1;
				//*save_write_loc = f0*0.6 + f1*0.1 + f2*0.1 + f3*0.1 + f4*0.1;
				
				//*save_write_loc = (*pf0)*0.6 + (*pf1)*0.1 + (*pf2)*0.1 + (*pf3)*0.1 + (*pf4)*0.1;
				f0 = *pf0;
				f1 = *pf1;
				f2 = *pf2;
				f3 = *pf3;
				f4 = *pf4;
				*save_write_loc++ = f0*0.6+f1*0.1+f2*0.1+f3*0.1+f4*0.1;
			/*	__nop();
				__nop();
				__nop();*/
				pf0++; pf1++; pf2++; pf3++; pf4++;
				/*f3 = d_matrix[x][y];*/
				
				//save_write_loc++;
			}
		}
		}
		#pragma omp parallel 
		{
		#pragma omp for schedule(static) nowait	
		for (int x = 0; x < DIM; x++) 
		{
			s_m = &d_matrix[x][0];
			s_r_m = &d_res_matrix[x][0];
					
			for (int y = 0; y < (DIM); y++)
			{
				//const int i_x = x;
				//d_matrix[x][y] = d_res_matrix[x][y];
				*s_m = *s_r_m;
				s_m++; s_r_m++;
				/**s_m = *s_r_m;
				s_m++; s_r_m++;
				*s_m = *s_r_m;
				s_m++; s_r_m++;
				*s_m = *s_r_m;
				s_m++; s_r_m++;
				*s_m = *s_r_m;
				s_m++; s_r_m++;
				*s_m = *s_r_m;
				s_m++; s_r_m++;
				*s_m = *s_r_m;
				s_m++; s_r_m++;
				*s_m = *s_r_m;
				s_m++; s_r_m++;*/
			}
			}
		}
	}
	STOP_TIMING_ND;

	//START_TIMING_MALLOC;
	//for (int iter = 0; iter < ITERATIONS; iter++)
	//{
	//	for (int x = 1; x < DIM-1; x++) // avoid boundary cases for this example
	//	{
	//		pf3 = &m_matrix[x][0];
	//		pf0 = &m_matrix[x][1];
	//		pf1 = &m_matrix[x-1][1];
	//		pf2 = &m_matrix[x+1][1];
	//		pf4 = &m_matrix[x][2];
	//		save_write_loc = &m_res_matrix[x][0];
	//		for (int y = 1; y < DIM-1; y++)
	//		{
	//			/*f0 = m_matrix[x][y];
	//			f1 = m_matrix[x-1][y];
	//			f2 = m_matrix[x+1][y];
	//			f3 = m_matrix[x][y-1];
	//			f4 = m_matrix[x][y+1];

	//			m_res_matrix[x][y] = f0*0.6 + f1*0.1 + f2*0.1 + f3*0.1 + f4*0.1;*/

	//			f0 = *pf0++;
	//			f1 = *pf1++;
	//			f2 = *pf2++;
	//			f3 = *pf3++;
	//			f4 = *pf4++;
	//			*save_write_loc++ = (f0)*0.6+(f1)*0.1+(f2)*0.1+(f3)*0.1+(f4)*0.1;
	//			/*__nop();
	//			__nop();
	//			__nop();*/
	//			//pf0++; pf1++; pf2++; pf3++; pf4++;
	//		}
	//	}
	//	for (int x = 1; x < DIM-1; x++)
	//	{
	//		/*for (int y = 1; y < DIM-1; y++)
	//		{
	//			m_matrix[x][y] = m_res_matrix[x][y];
	//		}*/
	//		s_m = &m_matrix[x][0];
	//		s_r_m = &m_res_matrix[x][0];
	//		for (int y = 0; y < (DIM/8); y+=8)
	//		{
	//			//d_matrix[x][y] = d_res_matrix[x][y];
	//			*s_m = *s_r_m++;
	//			s_m++; //s_r_m++;
	//			*s_m = *s_r_m++;
	//			s_m++; //s_r_m++;
	//			*s_m = *s_r_m++;
	//			s_m++; //s_r_m++;
	//			*s_m = *s_r_m++;
	//			s_m++; //s_r_m++;
	//			*s_m = *s_r_m++;
	//			s_m++; //s_r_m++;
	//			*s_m = *s_r_m++;
	//			s_m++; //s_r_m++;
	//			*s_m = *s_r_m++;
	//			s_m++; //s_r_m++;
	//			*s_m = *s_r_m++;
	//			s_m++; //s_r_m++;
	//		}
	//	}
	//}
	//STOP_TIMING_MALLOC;

	// Keep vectors from being deleted before end of function
	for (int x = 1; x < DIM-1; x++)
	{
		for (int y = 1; y < DIM-1; y++)
		{
			v_res_matrix[y][x] = v_matrix[x][y];
		}
	}

	// delete dynamic stuff
	for (int i = 0; i < DIM; i++)
	{
		delete [] d_matrix[i];
		delete [] d_res_matrix[i];
		free(m_matrix[i]);
		free(m_res_matrix[i]);
		
	}
	delete [] d_matrix;
	delete [] d_res_matrix;
	free(m_matrix);
	free(m_res_matrix);

	return 0;
}