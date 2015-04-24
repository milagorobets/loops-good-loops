#ifndef GPUALGOPIXELFLOW_H_
#define GPUALGOPIXELFLOW_H_

#include <map>

class GPU_UNOPTIMIZED
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

	float * m_ptr;

public:
	GPU_UNOPTIMIZED(); // default constructor
	void cpuAlgoPixelFlow_init(double matrixFlow[][4], double matrixWall[][4], double in_sourceLoc[]);
	void cpuAlgoPixelFlow(unsigned int num_iterations);
	void cpuAlgoPixelFlow_delete();
	double get_M0(int x, int y);
	void setMatrixWallLoc(int x, int y, int val);
};

#endif /* GPUALGOPIXELFLOW_H_ */
