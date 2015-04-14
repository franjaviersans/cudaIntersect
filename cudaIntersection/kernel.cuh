#ifndef __CUDACODE__  
#define __CUDACODE__

#include "Definitions.h"
#include "utils.h"
#include "timer.h"
#include "Transformation.h"

#include <vector>
#include <stdio.h>

using std::vector;



//#define ALLFALSE
//#define ALLTRUE
//#define ALLTEST

#define MAX_M 2000
#define MAX_N 1500


struct mat44
{
	float data[16];
};

class CUDA{
private:
	float3 * d_p1;
	float3 * d_p2;
	float3 * d_A;
	float4 * d_Normal;
	uint3 * d_B;
	mat44 * d_x, * h_x;
	unsigned int * d_inter;
	int sizeA, sizeB, sizeC, sizeN;
	float m_transX, m_transY, m_transZ; 

public:
	unsigned int block, threads, threadsxblock;
	__host__ void Init(float3 * A, uint3 * B, float4 * Normal, float3 * C, unsigned int sA, unsigned int sB, unsigned int sN, unsigned int sC);
	__host__ void InitOld(float3 * A, uint3 * B, float4 * Normal, float3 * C, unsigned int sA, unsigned int sB, unsigned int sN, unsigned int sC);
	__host__ void Destroy();
	__host__ bool CudaIntercept(float &time, float *out_trans, unsigned int * out_inter, unsigned int N, Transformation &t);
};


#endif