/*
 * cpuAlgoPixelFlow_v.h
 *
 *  Created on: 2015-04-01
 *      Author: cinnamon
 */

#ifndef CPUALGOPIXELFLOW_V_H_
#define CPUALGOPIXELFLOW_V_H_

#include <vector>

//namespace CPU_VECTOR
//{
//	void cpuAlgoPixelFlow_init(void);
//	void cpuAlgoPixelFlow(unsigned int num_iterations, double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[]);
//	void cpuAlgoPixelFlow_nextStep(void);
//	void cpuAlgoPixelFlow_updateSource(int t);
//	void cpuAlgoPixelFlow_delete();
//
//	extern int matrixWallLoc[MATRIX_DIM][MATRIX_DIM];
//	extern int entries;
//}

class CPU_VECTOR
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

	std::vector<std::vector<double> > m0, nm0;
	std::vector<std::vector<double> > m1, nm1;
	std::vector<std::vector<double> > m2, nm2;
	std::vector<std::vector<double> > m3, nm3;
	std::vector<std::vector<double> > W, WWall;

	double W00, W01, W02, W03, W10, W11, W12, W13, W20, W21, W22, W23, W30, W31, W32, W33;

	int matrixWallLoc[MATRIX_DIM][MATRIX_DIM];

public:
	CPU_VECTOR(); // default constructor
	void cpuAlgoPixelFlow_init(void);
	void cpuAlgoPixelFlow(unsigned int num_iterations, double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[]);
	void cpuAlgoPixelFlow_delete();
	double get_M0(int x, int y);
	void setMatrixWallLoc(int x, int y, int val);
};

#endif /* CPUALGOPIXELFLOW_V_H_ */
