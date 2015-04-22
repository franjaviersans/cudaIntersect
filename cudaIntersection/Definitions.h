//#ifndef DEF_H
//#define DEF_H

#define CUDA_CODE
#ifdef CUDA_CODE
	#include <cuda.h>
	#include <cuda_runtime.h>
	#include <cuda_runtime_api.h>
	#include <device_functions.h>
	#include <device_launch_parameters.h>

	#define GLM_FORCE_CUDA
	#include "kernel.cuh"
#endif


#include "include/glm/glm.hpp"
#include "include/glm/gtc/matrix_transform.hpp"
#include "include/glm/gtc/type_ptr.hpp"

#define GLFW_DLL
#include "include/GL/glew.h"

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



#define BUFFER_OFFSET(i) (reinterpret_cast<void*>(i))
#define WORLD_COORD_LOCATION 0
#define TEXTURE_COORD_LOCATION 1
#define NORMAL_COORD_LOCATION 2
#define COLOR_COORD_LOCATION 3

//#endif