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

#define N 800


class CUDA{
private:
	float3 * d_p1;
	float3 * d_p2;
	float3 * d_A;
	float4 * d_Normal;
	uint3 * d_B;
	unsigned int * d_triID;
	unsigned int * d_originID;
	unsigned int * d_destID;
	float * d_x;
	unsigned int * d_inter;
	int sizeA, sizeB, sizeC, sizeN;
	float m_transX, m_transY, m_transZ; 

public:
	unsigned int block, threads, threadsxblock;
	__host__ void Init(float3 * A, uint3 * B, float4 * Normal, float3 * C, unsigned int sA, unsigned int sB, unsigned int sN, unsigned int sC);
	__host__ void Destroy();
	__host__ bool CudaIntercept(float &time, Transformation *);
};


#endif