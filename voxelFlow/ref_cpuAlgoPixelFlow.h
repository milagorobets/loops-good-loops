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
	float f[4];
	float source;
	float sourceLoc[2];
	float src_amplitude, src_frequency;
	int coef;
	int entries;

	int WWAL_LENGTH ;
	int W_LENGTH;

	float ** m0, ** nm0; 
	float ** m1, ** nm1;
	float ** m2, ** nm2;
	float ** m3, ** nm3;
	float ** W, ** WWall;

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
	void cpuAlgoPixelFlow_init(float matrixFlow[][4], float matrixWall[][4], float in_sourceLoc[]);
	void cpuAlgoPixelFlow(unsigned int num_iterations);
	void cpuAlgoPixelFlow_delete();
	double get_M0(int x, int y);
	void setMatrixWallLoc(int x, int y, int val);
};

#endif /* REF_CPUALGOPIXELFLOW_H_ */
