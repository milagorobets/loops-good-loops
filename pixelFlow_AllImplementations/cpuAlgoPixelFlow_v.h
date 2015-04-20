/*
 * cpuAlgoPixelFlow_v.h
 *
 *  Created on: 2015-04-01
 *      Author: cinnamon
 */

#ifndef CPUALGOPIXELFLOW_V_H_
#define CPUALGOPIXELFLOW_V_H_

namespace CPU_VECTOR
{
	void cpuAlgoPixelFlow_init(void);
	void cpuAlgoPixelFlow(unsigned int num_iterations, double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[]);
	void cpuAlgoPixelFlow_nextStep(void);
	void cpuAlgoPixelFlow_updateSource(int t);
	void cpuAlgoPixelFlow_delete();

	extern int matrixWallLoc[MATRIX_DIM][MATRIX_DIM];
	extern int entries;
}

#endif /* CPUALGOPIXELFLOW_V_H_ */
