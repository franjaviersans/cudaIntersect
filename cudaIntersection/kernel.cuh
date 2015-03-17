#ifndef __CUDACODE__  
#define __CUDACODE__

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "utils.h"
#include "timer.h"

#include <stdio.h>


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


__device__ bool ray_triangle( const float V1[3],  // Triangle vertices
                           const float V2[3],
                           const float V3[3],
                           const float O[3],  //Ray origin
                           const float D[3]  //Ray direction
						   );


__global__ void Intercept(const float * const p1, const float * const p2,
			   const float * const A, const unsigned int * const B,
			   const unsigned int sizeC, const unsigned int sizeA,
			   const unsigned int sizeB);


void CudaIntercept(float &time, float * A, unsigned int  * B, unsigned int sizeA, unsigned int sizeB);


#endif