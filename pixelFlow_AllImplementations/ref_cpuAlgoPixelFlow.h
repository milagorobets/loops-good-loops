/*
 * cpuAlgoPixelFlow.h
 *
 *  Created on: 2015-02-20
 *      Author: cinnamon
 */

#ifndef REF_CPUALGOPIXELFLOW_H_
#define REF_CPUALGOPIXELFLOW_H_

namespace REFERENCE_SPACE
{
	void cpuAlgoPixelFlow_init(void);
	void cpuAlgoPixelFlow(unsigned int num_iterations, double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[]);
	void cpuAlgoPixelFlow_nextStep(void);
	void cpuAlgoPixelFlow_updateSource(int t);
	void cpuAlgoPixelFlow_delete();
	double get_M0(int x, int y);

	extern int matrixWallLoc[MATRIX_DIM][MATRIX_DIM];
	extern int entries;
}
#endif /* REF_CPUALGOPIXELFLOW_H_ */
