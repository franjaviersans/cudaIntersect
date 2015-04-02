

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