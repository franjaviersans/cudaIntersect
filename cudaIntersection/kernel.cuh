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


#define M 2000
#define N 200

#define EPSILON 0.000001
#define CROSS(dest, v1, v2) \
	dest[0] = v1[1]*v2[2] - v1[2]*v2[1]; \
	dest[1] = v1[2]*v2[0] - v1[0]*v2[2]; \
	dest[2] = v1[0]*v2[1] - v1[1]*v2[0];
#define DOT(v1, v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest, v1, v2) \
	dest[0] = v1[0] - v2[0]; \
	dest[1] = v1[1] - v2[1]; \
	dest[2] = v1[2] - v2[2]; 

#define MULT(dest, mat,p) \
	dest[0] = mat[0] * p[0] + mat[1] * p[1] + mat[2] * p[2] + mat[3] * 1.0f; \
	dest[1] = mat[4] * p[0] + mat[5] * p[1] + mat[6] * p[2] + mat[7] * 1.0f; \
	dest[2] = mat[8] * p[0] + mat[9] * p[1] + mat[10] * p[2] + mat[11] * 1.0f; 


class CUDA{
float * d_p1;
float * d_p2;
float * d_A;
unsigned int * d_B;
float * d_x;
bool * d_inter;
int sizeA, sizeB, sizeC;

public:
__host__ void Init(float * A, unsigned int  * B, float *C, unsigned int sA, unsigned int sB, unsigned int sC);
__host__ void Destroy();
__host__ bool CudaIntercept(float &time);
};


__device__ bool ray_triangle( const float V1[3],  // Triangle vertices
                           const float V2[3],
                           const float V3[3],
                           const float O[3],  //Ray origin
                           const float D[3]  //Ray direction
						   );

__device__ bool Test2(float v[3], unsigned int i);


__global__ void Intercept(const float * const p1, const float * const p2,
			   const float * const A, const unsigned int * const B,
			   const unsigned int sizeC, const unsigned int sizeA,
			   const unsigned int sizeB, 
			   const float * const x,
			   bool * globalinter);



#endif