#include "kernel.cuh"

__device__ bool ray_triangle( const float V1[3],  // Triangle vertices
                           const float V2[3],
                           const float V3[3],
                           const float O[3],  //Ray origin
                           const float D[3]  //Ray direction
						   )
{

	#ifdef ALLFALSE
		return false;
	#endif

	#ifdef ALLTRUE
		return true;
	#endif

	float e1[3], e2[3];  //Edge1, Edge2
	float P[3], Q[3], T[3];
	float det, inv_det, u, v;
	float t;
 
	//Find vectors for two edges sharing V1
	SUB(e1, V2, V1);
	SUB(e2, V3, V1);
	//Begin calculating determinant - also used to calculate u parameter
	CROSS(P, D, e2);
	//if determinant is near zero, ray lies in plane of triangle
	det = DOT(e1, P);
	//NOT CULLING
	if(det > -EPSILON && det < EPSILON) return false;
	inv_det = 1.f / det;
 
	//calculate distance from V1 to ray origin
	SUB(T, O, V1);
 
	//Calculate u parameter and test bound
	u = DOT(T, P) * inv_det;
	//The intersection lies outside of the triangle
	if(u < 0.f || u > 1.f) return false;
 
	//Prepare to test v parameter
	CROSS(Q, T, e1);
 
	//Calculate V parameter and test bound
	v = DOT(D, Q) * inv_det;
	//The intersection lies outside of the triangle
	if(v < 0.f || u + v  > 1.f) return false;
 
	t = DOT(e2, Q) * inv_det;
 
	if(t > EPSILON && t < 1.0f) { //ray intersection
	return true;
	}
 
	// No hit, no win
	return false;
	
}


__device__ bool Test2(float v[3], unsigned int i){

	#ifdef ALLFALSE
		return false;
	#endif

	#ifdef ALLTRUE
		return true;
	#endif

	return v[0] * v[0] + v[1] * v[1] < v[2] * v[2] + i * i;

}

__global__ void Intercept(const float * const p1, const float * const p2,
			   const float * const A, const unsigned int * const B,
			   const unsigned int sizeC, const unsigned int sizeA,
			   const unsigned int sizeB, 
			   const float * const x,
			   bool * const globalinter)
{

	//Shared memory declaration
	extern __shared__ char buffer[];

	float * dir = (float *)&buffer[0];
    float * origin = (float *)&buffer[sizeC * 3 * sizeof(float)];
	float * lt = (float *)&buffer[sizeC * 3 * sizeof(float) + 3 * sizeof(float)];
	bool * sharedinter = (bool *)&buffer[sizeC * 3 * sizeof(float) + 3 * sizeof(float) + 16 *sizeof(float)];

	unsigned int tid = threadIdx.x;
	unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
	float v0[3], v1[3], v2[3], vaux[3];
	bool inter = false;
	
	
	if(tid == 0)
	{
		*sharedinter = false;
		vaux[0] = p1[0];
		vaux[1] = p1[1];
		vaux[2] = p1[2];
		MULT(origin, lt, vaux);
	}
	if(tid < 16) lt[tid] = x[tid];
	__syncthreads();

	//Copy all the data of C
	if(tid < sizeC)
	{
		dir[tid * 3] = p2[tid * 3] - origin[0];
		dir[tid * 3 + 1] = p2[tid * 3 + 1] - origin[1];
		dir[tid * 3 + 2] = p2[tid * 3 + 2] - origin[2];
	}
	__syncthreads();

	

	if(globalTid < sizeB){
		

		//Point 0
		vaux[0] = A[B[globalTid * 3] * 3];
		vaux[1] = A[B[globalTid * 3] * 3 + 1];
		vaux[2] = A[B[globalTid * 3] * 3 + 2];

		MULT(v0, lt, vaux);

		//Point 1
		vaux[0] = A[B[globalTid * 3 + 1] * 3];
		vaux[1] = A[B[globalTid * 3 + 1] * 3 + 1];
		vaux[2] = A[B[globalTid * 3 + 1] * 3 + 2];

		MULT(v1, lt, vaux);

		//Point 2
		vaux[0] = A[B[globalTid * 3 + 2] * 3];
		vaux[1] = A[B[globalTid * 3 + 2] * 3 + 1];
		vaux[2] = A[B[globalTid * 3 + 2] * 3 + 2];
	
		MULT(v2, lt, vaux);
		

		//First test. Ray-Triangle Intersection
		unsigned int i;
		for(i=0; i < sizeC && !*globalinter;++i)
		{
			inter = ray_triangle(v0, v1, v2, origin, &(dir[i * 3]));
			#ifdef ALLTEST
				if(inter) *globalinter = false;
			#else
				if(inter) *globalinter = true;
			#endif
		}
		//if(globalTid == 0)printf("%d \n",i);
	}

}


__global__ void SecondTest(	const float * const A, const unsigned int sizeA, bool * const globalinter)
{
	unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x;

	bool inter = false;

	float v[3];
	
	if(globalTid < sizeA)
	{
		v[0] = A[globalTid * 3];
		v[1] = A[globalTid * 3 + 1];
		v[2] = A[globalTid * 3 + 2];

		//Second Test
		unsigned int j;
		for(j=0;j < N && !*globalinter;++j)
		{
			inter = Test2(v, j);
			#ifdef ALLTEST
				if(inter) *globalinter = false;
			#else
				if(inter) *globalinter = true;
			#endif
		}
		//if(globalTid == 0)printf("%d \n",j);
	}
}

bool CUDA::CudaIntercept(float &time){
	bool h_inter = false;

	checkCudaErrors(cudaMemcpy(d_inter, &h_inter,  sizeof(bool), cudaMemcpyHostToDevice));

	//Each thread for each triangle
	dim3 BlockDim(128, 1, 1);
	dim3 GridDim((sizeB + BlockDim.x)/BlockDim.x, 1, 1);
	//printf("(1:%d)", h_inter);
	//printf("Num Faces: %d\n ThreadsxBlock: %d\n Num Blocks: %d\n", sizeB, BlockDim.x, GridDim.x);

	//First test with timer
	GpuTimer timer;
	timer.Start();
	Intercept<<< GridDim, BlockDim, 100 * 3 * sizeof(float) + 3 * sizeof(float) + sizeof(bool) + 16 *sizeof(float) >>>(d_p1, d_p2, d_A, d_B, 100, sizeA, sizeB, d_x, d_inter);
	timer.Stop();

	time += timer.Elapsed();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(&h_inter, d_inter,  sizeof(bool), cudaMemcpyDeviceToHost));
	//printf("(2:%d)", h_inter);
	if(!h_inter){

		//One thread for each point in the surface
		dim3 BlockDim2(128, 1, 1);
		dim3 GridDim2((sizeA + BlockDim.x)/BlockDim.x, 1, 1);

		//Second test with timer
		timer.Start();
		SecondTest<<< GridDim2, BlockDim2>>>(d_A, sizeA, d_inter);
		timer.Stop();

		time += timer.Elapsed();
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaMemcpy(&h_inter, d_inter,  sizeof(bool), cudaMemcpyDeviceToHost));
		//printf("(3:%d)", h_inter);
	}
	

	

	return h_inter;

}


__host__ void CUDA::Init(float * A, unsigned int  * B, float *C, unsigned int sA, unsigned int sB, unsigned int sC){
	float center[] = {0.0f,0.0f,0.0f};
	float transform[] = {	1.0f,0.0f,0.0f,0.0f,
							0.0f,1.0f,0.0f,0.0f,
							0.0f,0.0f,1.0f,0.0f,
							0.0f,0.0f,0.0f,1.0f};

	sizeA = sA;
	sizeB = sB;
	sizeC = sC;

	checkCudaErrors(cudaSetDevice(0));

	//Allocate memory on the GPU

	checkCudaErrors(cudaMalloc((void**)&d_p1, 3 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_p2, sizeC * 3 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_A, sizeA * 3 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_B, sizeB * 3 * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_x, 16 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_inter, sizeof(bool)));
	
	//Send information to the GPU
	checkCudaErrors(cudaMemcpy(d_p1, center, 3 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_p2, C, sizeC * 3 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_A, A, sizeA * 3 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, B, sizeB * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_x, transform,  16 * sizeof(float), cudaMemcpyHostToDevice));
	

}

__host__ void CUDA::Destroy(){
	//Free memory
	checkCudaErrors(cudaFree(d_p1));
	checkCudaErrors(cudaFree(d_p2));
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(d_inter));
}