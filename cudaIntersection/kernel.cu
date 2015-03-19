#include "kernel.cuh"

__device__ bool ray_triangle( const float3 V1,  // Triangle vertices
                           const float3 V2,
                           const float3 V3,
                           const float3 O,  //Ray origin
                           const float3 D  //Ray direction
						   )
{

	#ifdef ALLFALSE
		return false;
	#endif

	#ifdef ALLTRUE
		return true;
	#endif

	float3 e1, e2;  //Edge1, Edge2
	float3 P, Q, T;
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


__device__ bool Test2(float3 v, unsigned int i){

	#ifdef ALLFALSE
		return false;
	#endif

	#ifdef ALLTRUE
		return true;
	#endif

	return v.x * v.x + v.y * v.y < v.z * v.z + i * i;

}

__global__ void Intercept(const float3 * const p1, const float3 * const p2,
			   const float3 * const A, const uint3 * const B,
			   const unsigned int sizeC, const unsigned int sizeA,
			   const unsigned int sizeB, 
			   const float * const x,
			   bool * const globalinter)
{

	//Shared memory declaration
	extern __shared__ char buffer[];

	float3 * dir = (float3 *)&buffer[0];
    float3 * origin = (float3 *)&buffer[sizeC * sizeof(float3)];
	float * lt = (float *)&buffer[sizeC * sizeof(float3) + sizeof(float3)];
	bool * sharedinter = (bool *)&buffer[sizeC * sizeof(float3) + sizeof(float3) + 16 *sizeof(float)];

	unsigned int tid = threadIdx.x;
	unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
	float3 v0, v1, v2, vaux;
	bool inter = false;
	
	
	if(tid == 0)
	{
		*sharedinter = false;
		vaux = *p1;
		MULT((*origin), lt, vaux);
	}
	if(tid < 16) lt[tid] = x[tid];
	__syncthreads();

	//Copy all the data of C
	if(tid < sizeC)
	{
		v0 = p2[tid];

		MULT(vaux, lt, v0);


		dir[tid].x = vaux.x - (*origin).x;
		dir[tid].y = vaux.y - (*origin).y;
		dir[tid].z = vaux.z - (*origin).z;
		
		
	}
	__syncthreads();

	

	if(globalTid < sizeB){
		
		uint3 id = B[globalTid];
		//Point 0
		v0 = A[id.x];

		//Point 1
		v1 = A[id.y];

		//Point 2
		v2 = A[id.z];

		//First test. Ray-Triangle Intersection
		unsigned int i;
		for(i=0; i < sizeC && !*globalinter;++i)
		{
			inter = ray_triangle(v0, v1, v2, (*origin), dir[i]);
			#ifdef ALLTEST
				if(inter) *globalinter = false;
			#else
				if(inter) *globalinter = true;
			#endif
		}
		//if(globalTid == 0)printf("%d \n",i);
	}

}


__global__ void SecondTest(	const float3 * const A, const unsigned int sizeA, bool * const globalinter)
{
	unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x;

	bool inter = false;

	float3 v;
	
	if(globalTid < sizeA)
	{
		v = A[globalTid];

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

	//Generate a random transform with scaling and translating
	float transform[] = {	1.0f,0.0f,0.0f,0.0f,
							0.0f,1.0f,0.0f,0.0f,
							0.0f,0.0f,1.0f,0.0f,
							0.0f,0.0f,0.0f,1.0f};
	//Translate
	transform[3] = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f;
	transform[7] = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f;
	transform[11] = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f;

	//Scaling
	transform[0] = 
	transform[5] = 
	transform[10] = (rand() % RAND_MAX) / float(RAND_MAX);

	checkCudaErrors(cudaMemcpy(d_x, transform,  16 * sizeof(float), cudaMemcpyHostToDevice));
	

	checkCudaErrors(cudaMemcpy(d_inter, &h_inter,  sizeof(bool), cudaMemcpyHostToDevice));

	//Each thread for each triangle
	dim3 BlockDim(128, 1, 1);
	dim3 GridDim((sizeB + BlockDim.x)/BlockDim.x, 1, 1);
	//printf("(1:%d)", h_inter);
	//printf("Num Faces: %d\n ThreadsxBlock: %d\n Num Blocks: %d\n", sizeB, BlockDim.x, GridDim.x);

	//First test with timer
	GpuTimer timer;
	timer.Start();
	Intercept<<< GridDim, BlockDim, 100 * sizeof(float3) + sizeof(float3) + sizeof(bool) + 16 *sizeof(float) >>>(d_p1, d_p2, d_A, d_B, 100, sizeA, sizeB, d_x, d_inter);
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


__host__ void CUDA::Init(float3 * A, uint3  * B, float3 *C, unsigned int sA, unsigned int sB, unsigned int sC){
	float center[] = {0.0f,0.0f,0.0f};

	sizeA = sA;
	sizeB = sB;
	sizeC = sC;

	/* initialize random seed: */
	srand (time(NULL));

	checkCudaErrors(cudaSetDevice(0));

	//Allocate memory on the GPU

	checkCudaErrors(cudaMalloc((void**)&d_p1, sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_p2, sizeC * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_A, sizeA * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_B, sizeB * sizeof(uint3)));
	checkCudaErrors(cudaMalloc((void**)&d_x, 16 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_inter, sizeof(bool)));
	
	//Send information to the GPU
	checkCudaErrors(cudaMemcpy(d_p1, center, sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_p2, C, sizeC * sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_A, A, sizeA * sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, B, sizeB * sizeof(uint3), cudaMemcpyHostToDevice));

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