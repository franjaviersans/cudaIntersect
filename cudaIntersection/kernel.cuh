#ifndef __CUDACODE__  
#define __CUDACODE__

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "utils.h"
#include "timer.h"

#include <stdio.h>



//#define ALLFALSE
//#define ALLTRUE
//#define ALLTEST


#define M 2000
#define N 1500

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
	dest.x = mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3] * 1.0f; \
	dest.y = mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7] * 1.0f; \
	dest.z = mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11] * 1.0f; 


class CUDA{
float3 * d_p1;
float3 * d_p2;
float3 * d_A;
uint3 * d_B;
float * d_x;
bool * d_inter;
int sizeA, sizeB, sizeC;

public:
__host__ void Init(float3 * A, uint3 * B, float3 * C, unsigned int sA, unsigned int sB, unsigned int sC);
__host__ void Destroy();
__host__ bool CudaIntercept(float &time);
};


__device__ bool ray_triangle( float3 V1,  // Triangle vertices
                           float3 V2,
                           float3 V3,
                           float3 O,  //Ray origin
                           float3 D  //Ray direction
						   );

__device__ bool Test2(float v[3], unsigned int i);


__global__ void Intercept(const float * const p1, const float3 * const p2,
			   const float3 * const A, const uint3 * const B,
			   const unsigned int sizeC, const unsigned int sizeA,
			   const unsigned int sizeB, 
			   const float * const x,
			   bool * globalinter);



#endif