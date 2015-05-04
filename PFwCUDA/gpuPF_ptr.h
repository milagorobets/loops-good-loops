#ifndef GPUPF_H_
#define GPUPF_H_

#include <map>

class GPU_PTR
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

	float * m_ptr;

public:
	GPU_PTR(); // default constructor
	void cpuAlgoPixelFlow_init(float matrixFlow[][4], float matrixWall[][4], float in_sourceLoc[]);
	void cpuAlgoPixelFlow(unsigned int num_iterations);
	void cpuAlgoPixelFlow_delete();
	double get_M0(int x, int y);
	void setMatrixWallLoc(int x, int y, int val);
	void setupDisplay(void);
};

#endif /* GPUPF_H_ */
