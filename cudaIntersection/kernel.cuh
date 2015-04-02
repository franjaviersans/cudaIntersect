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

#define EPSILON 0.000001
#define CROSS(dest, v1, v2) \
	dest.x = v1.y*v2.z - v1.z*v2.y; \
	dest.y = v1.z*v2.x - v1.x*v2.z; \
	dest.z = v1.x*v2.y - v1.y*v2.x; 
#define DOT(v1, v2) (v1.x*v2.x+v1.y*v2.y+v1.z*v2.z)
#define SUB(dest, v1, v2) \
	dest.x = v1.x - v2.x; \
	dest.y = v1.y - v2.y; \
	dest.z = v1.z - v2.z; 
#define MULT(dest, mat,p) \
	dest.x = mat[0] * p.x + mat[4] * p.y + mat[8] * p.z + mat[12] * 1.0f; \
	dest.y = mat[1] * p.x + mat[5] * p.y + mat[9] * p.z + mat[13] * 1.0f; \
	dest.z = mat[2] * p.x + mat[6] * p.y + mat[10] * p.z + mat[14] * 1.0f; 


class CUDA{
private:
	float3 * d_p1;
	float3 * d_p2;
	float3 * d_A;
	uint3 * d_B;
	unsigned int * d_triID;
	unsigned int * d_originID;
	unsigned int * d_destID;
	float * d_x;
	unsigned int * d_inter;
	int sizeA, sizeB, sizeC;

public:
	unsigned int block, threads, threadsxblock;
	__host__ void Init(float3 * A, uint3 * B, float3 * C, unsigned int sA, unsigned int sB, unsigned int sC);
	__host__ void Destroy();
	__host__ bool CudaIntercept(float &time, Transformation *);
};


#endif