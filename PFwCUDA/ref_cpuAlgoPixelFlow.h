/*
 * cpuAlgoPixelFlow.h
 *
 *  Created on: 2015-02-20
 *      Author: cinnamon
 */

#ifndef REF_CPUALGOPIXELFLOW_H_
#define REF_CPUALGOPIXELFLOW_H_

#include <map>

class CPU_REFERENCE
{
private:
	void cpuAlgoPixelFlow_nextStep(void);
	void cpuAlgoPixelFlow_updateSource(int t);
	double f[4];
	double source;
	double sourceLoc[2];
	double src_amplitude, src_frequency;
	int coef;
	int entries;

	int WWAL_LENGTH ;
	int W_LENGTH;

	double ** m0, ** nm0; 
	double ** m1, ** nm1;
	double ** m2, ** nm2;
	double ** m3, ** nm3;
	double ** W, ** WWall;

#if (WALL_MEMORY==MEM_STACK)
	bool matrixWallLoc[MATRIX_DIM][MATRIX_DIM];
#elif (WALL_MEMORY==MEM_MAP)
	std::map<int,std::map<int,bool> > mapWallLoc;
#elif (WALL_MEMORY==MEM_HEAP)
	bool ** heapWallLoc;
#else
	#error("Select valid WALL_MEMORY in common.h")
#endif

public:
	CPU_REFERENCE(); // default constructor
	void cpuAlgoPixelFlow_init(double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[]);
	void cpuAlgoPixelFlow(unsigned int num_iterations);
	void cpuAlgoPixelFlow_delete();
	double get_M0(int x, int y);
	void setMatrixWallLoc(int x, int y, int val);
};

#endif /* REF_CPUALGOPIXELFLOW_H_ */