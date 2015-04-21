/*
 * cpuAlgoPixelFlow.h
 *
 *  Created on: 2015-02-20
 *      Author: cinnamon
 */

#ifndef REF_CPUALGOPIXELFLOW_H_
#define REF_CPUALGOPIXELFLOW_H_

//namespace REFERENCE_SPACE
//{
//	void cpuAlgoPixelFlow_init(void);
//	void cpuAlgoPixelFlow(unsigned int num_iterations, double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[]);
//	void cpuAlgoPixelFlow_nextStep(void);
//	void cpuAlgoPixelFlow_updateSource(int t);
//	void cpuAlgoPixelFlow_delete();
//	double get_M0(int x, int y);
//
//	extern int matrixWallLoc[MATRIX_DIM][MATRIX_DIM];
//	extern int entries;
//}

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

	double ** m0, ** nm0; // try the linear allocation method for timing comparison
	double ** m1, ** nm1;
	double ** m2, ** nm2;
	double ** m3, ** nm3;
	double ** W, ** WWall;

	int matrixWallLoc[MATRIX_DIM][MATRIX_DIM];

public:
	CPU_REFERENCE(); // default constructor
	void cpuAlgoPixelFlow_init(void);
	void cpuAlgoPixelFlow(unsigned int num_iterations, double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[]);
	void cpuAlgoPixelFlow_delete();
	double get_M0(int x, int y);
	void setMatrixWallLoc(int x, int y, int val);
};
#endif /* REF_CPUALGOPIXELFLOW_H_ */
