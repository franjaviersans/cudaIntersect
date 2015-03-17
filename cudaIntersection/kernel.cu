#include "kernel.cuh"

__device__ bool ray_triangle( const float V1[3],  // Triangle vertices
                           const float V2[3],
                           const float V3[3],
                           const float O[3],  //Ray origin
                           const float D[3]  //Ray direction
						   )
{
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
  if(det > -EPSILON && det < EPSILON) return 0;
  inv_det = 1.f / det;
 
  //calculate distance from V1 to ray origin
  SUB(T, O, V1);
 
  //Calculate u parameter and test bound
  u = DOT(T, P) * inv_det;
  //The intersection lies outside of the triangle
  if(u < 0.f || u > 1.f) return 0;
 
  //Prepare to test v parameter
  CROSS(Q, T, e1);
 
  //Calculate V parameter and test bound
  v = DOT(D, Q) * inv_det;
  //The intersection lies outside of the triangle
  if(v < 0.f || u + v  > 1.f) return 0;
 
  t = DOT(e2, Q) * inv_det;
 
  if(t > EPSILON && t < 1.0f) { //ray intersection
    return 1;
  }
 
  // No hit, no win
  return 0;
}


__global__ void Intercept(const float * const p1, const float * const p2,
			   const float * const A, const unsigned int * const B,
			   const unsigned int sizeC, const unsigned int sizeA,
			   const unsigned int sizeB)
{

	//Shared memory declaration
	extern __shared__ float buffer[];

	float * dir = (float *)&buffer[0];
    float * origin = (float *)&buffer[sizeC * 3];

	unsigned int tid = threadIdx.x;
	unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
	float v0[3], v1[3], v2[3];
	bool inter = false;
	
	if(tid < 3)
	{
		origin[tid] = p1[tid];
	}
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
		v0[0] = A[B[globalTid * 3] * 3];
		v0[1] = A[B[globalTid * 3] * 3 + 1];
		v0[2] = A[B[globalTid * 3] * 3 + 2];

		//Point 1
		v1[0] = A[B[globalTid * 3 + 1] * 3];
		v1[1] = A[B[globalTid * 3 + 1] * 3 + 1];
		v1[2] = A[B[globalTid * 3 + 1] * 3 + 2];

		//Point 2
		v2[0] = A[B[globalTid * 3 + 2] * 3];
		v2[1] = A[B[globalTid * 3 + 2] * 3 + 1];
		v2[2] = A[B[globalTid * 3 + 2] * 3 + 2];
	
		for(unsigned int i=0; i <sizeC && !inter;++i)
		{
			inter = ray_triangle(v0, v1, v2, origin, &dir[i * 3]);
		}
	}

}


void CudaIntercept(float &time, float * A, unsigned int  * B, unsigned int sizeA, unsigned int sizeB){

	float * d_p1;
	float * d_p2;
	float * d_A;
	unsigned int * d_B;
	unsigned int sizeC = 100;


	checkCudaErrors(cudaSetDevice(0));

	//Allocate memory on the GPU

	checkCudaErrors(cudaMalloc((void**)&d_p1, 3 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_p2, sizeC * 3 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_A, sizeA * 3 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_B, sizeB * 3 * sizeof(unsigned int)));
	
	//Send information to the GPU


	checkCudaErrors(cudaMemcpy(d_A, A, sizeA * 3 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, B, sizeB * 3 * sizeof(unsigned int), cudaMemcpyHostToDevice));



	//checkCudaErrors(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
	dim3 BlockDim(512, 1, 1);
	dim3 GridDim((sizeB + BlockDim.x)/BlockDim.x, 1, 1);

	//printf("Num Faces: %d\n ThreadsxBlock: %d\n Num Blocks: %d\n", sizeB, BlockDim.x, GridDim.x);

	GpuTimer timer;

	timer.Start();
	Intercept<<< GridDim, BlockDim, sizeC * 3 * sizeof(float) >>>(d_p1, d_p2, d_A, d_B, sizeC, sizeA, sizeB);
	timer.Stop();

	time += timer.Elapsed();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	//Free memory
	checkCudaErrors(cudaFree(d_p1));
	checkCudaErrors(cudaFree(d_p2));
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));



}
