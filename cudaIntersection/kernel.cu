#include "kernel.cuh"


//Ray-Triangle function to calculate the intersection
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
		//We only consider the [0.0,1.0] interval because is the distance between the origin and the point in the surface C
		return true;
	}
 
	// No hit, no win
	return false;
}

//A simple geometry test to be done in the points of A
//For now this is only a dommy test
__device__ bool Test2(float3 v, unsigned int i){

	#ifdef ALLFALSE
		return false;
	#endif

	#ifdef ALLTRUE
		return true;
	#endif

	//return v.x * v.x + v.y * v.y < v.z * v.z + i * i;
	return false;
}

__global__ void Intercept(const float3 * const p1, const float3 * const p2,
			   const float3 * const A, const uint3 * const B,
			   const unsigned int sizeC, const unsigned int sizeA,
			   const unsigned int sizeB, 
			   const float * const x,
			   unsigned int * const globalinter, unsigned int * const triID, unsigned int * const originID, unsigned int * const destID)
{

	//Shared memory declaration
	extern __shared__ char buffer[];

	//All point of C will be stored in the array dir in shared memory
	float3 * dir = (float3 *)&buffer[0];
    float3 * origin = (float3 *)&buffer[sizeC * sizeof(float3)]; //The origin will be shared to
	float * lt = (float *)&buffer[sizeC * sizeof(float3) + sizeof(float3)]; //The shared transformation matrix
	bool * sharedinter = (bool *)&buffer[sizeC * sizeof(float3) + sizeof(float3) + 16 *sizeof(float)]; //A boolean to know if the test has to stop

	//Id of the thread within a block and within the grid
	unsigned int tid = threadIdx.x;
	unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x;

	//Auxiliar variables
	float3 v0, v1, v2, vaux;
	bool inter = false;
	
	//if it is the first thread of the block
	if(tid == 0)
	{
		//Set global boolean to false (there has been no ray-triangle intersection
		*sharedinter = false;
		vaux = *p1; //Copy the data of the origin of the ray
		MULT((*origin), x, vaux); //Transfor the point. This is the only transformation done with global transformation data!!!
	}
	if(tid < 16) lt[tid] = x[tid]; //16 values of the 4x4 transformation matrix
	__syncthreads(); //Wait to all the threads in the block

	//Copy all the data of C
	unsigned int temp = tid;
	while(temp < sizeC)
	{
		//Copy a point of C to local data
		v0 = p2[temp];

		//Transform the point 
		MULT(vaux, lt, v0);

		//store the direction of the ray in shared memory x(p2) - x(p1)
		dir[temp].x = vaux.x - (*origin).x;
		dir[temp].y = vaux.y - (*origin).y;
		dir[temp].z = vaux.z - (*origin).z;

		temp += blockDim.x;
	}
	__syncthreads(); //Wait to all the threads in the block

	
	//Each thread works with one triangle in the surface (A, B)
	if(globalTid < sizeB){
		uint3 id = B[globalTid];

		//Store the points of the triangles in local memory
		v0 = A[id.x]; //Point 0
		v1 = A[id.y]; //Point 1
		v2 = A[id.z]; //Point 2

		//First test. Ray-Triangle Intersection
		unsigned int i;
		for(i=1; i < sizeC && *globalinter == 0;++i)  //For all the points in C do the intersection test
		{
			inter = ray_triangle(v0, v1, v2, (*origin), dir[i]); //Intersection function with the 3 points of the triangle, the origin, and the ith direction
			#ifdef ALLTEST
				if(inter) *globalinter = 0;
			#else
				if(inter)
				{
					//Safe the intersection line and triangle
					//We use atomicCas to only store one value
					if( 0 == atomicCAS(globalinter, 0, 1)) //If and intersection has ocurred, set the global intersection flag to true
					{	
						*triID = globalTid;
						*originID = 0;
						*destID = i; 
					}

				}
			#endif
		}
	}

}

//Kernel to do the second test
//Test in every point of the surface (A, B)
__global__ void SecondTest(	const float3 * const A, const unsigned int sizeA, unsigned int * const globalinter)
{
	//Id of the thread in the grid
	unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x;

	bool inter = false;

	float3 v;
	
	if(globalTid < sizeA)
	{
		v = A[globalTid]; //Copy the data of the point into local memory

		//Second Test
		unsigned int j;
		for(j=0;j < N && *globalinter == 0;++j) //Do the tests
		{
			inter = Test2(v, j);
			#ifdef ALLTEST
				if(inter) *globalinter = 0;
			#else
				if(inter) *globalinter = 1;
			#endif
		}
		//if(globalTid == 0)printf("%d \n",j);
	}
}


__global__ void Inside(const float3 * const p1,
			   const float4 * const normal, const unsigned int sizeNormal, 
			   const float transX, const float transY, const float transZ, unsigned int * const globalinter)
{

	//Shared memory declaration
	extern __shared__ char buffer[];

	//All point of C will be stored in the array dir in shared memory
	float3 * center = (float3 *)&buffer[0]; //The origin will be shared too

	//Id of the thread within a block and within the grid
	unsigned int tid = threadIdx.x;
	unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x;

	//Auxiliar variables
	float3 vaux;
	float4 n;

	//if it is the first thread of the block
	if(tid == 0)
	{
		vaux = *p1; //Copy the data of the origin of the ray
		(*center).x = vaux.x + transX;
		(*center).y = vaux.y + transY;
		(*center).z = vaux.z + transZ;
	}
	__syncthreads(); //Wait to all the threads in the block

	if(globalTid < sizeNormal){
		n = normal[globalTid];
		float res = DOT(n, (*center));
		if(res + n.w > 0.0f)
		{
			*globalinter = 1;
		}
	}
}

bool CUDA::CudaIntercept(float &time, Transformation *t){
	unsigned int h_inter = 1;
	GpuTimer timer;

	//Generate a random transform with scaling, translating and rotating

	//Check if center is inside surface (A, B)
	while(h_inter != 0)
	{
		h_inter = 0;
		checkCudaErrors(cudaMemcpy(d_inter, &h_inter,  sizeof(unsigned int), cudaMemcpyHostToDevice));

		t->m_fTranslationx = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f + m_transX;
		t->m_fTranslationy = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f + m_transY;
		t->m_fTranslationz = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f + m_transZ;

		//Each thread for each triangle
		dim3 BlockDimAux(threadsxblock, 1, 1); //128 threads per block
		dim3 GridDimAux(block, 1, 1); 

		//We test if "center" is inside surface (A, B)
		timer.Start();
		Inside<<< GridDimAux, BlockDimAux, sizeof(float3)>>>(d_p1, d_Normal, sizeN, t->m_fTranslationx, t->m_fTranslationy, t->m_fTranslationz, d_inter);
		timer.Stop();

		time += timer.Elapsed();
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); //Check for errors
		
		checkCudaErrors(cudaMemcpy(&h_inter, d_inter,  sizeof(unsigned int), cudaMemcpyDeviceToHost)); //Copy the GPU global intersection variable to know if there was an intersection
	}


	t->m_fScalar = (rand() % RAND_MAX) / float(RAND_MAX * 2.0f) + 0.5f;
	

	glm::vec3 rotation_angle = glm::normalize(glm::vec3((rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f, (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f, (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f));
	t->m_fRotationAngle = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f;
	t->m_fRotationVectorx = rotation_angle.x;
	t->m_fRotationVectory = rotation_angle.y;
	t->m_fRotationVectorz = rotation_angle.z;
	

	//Generate quaternion
	glm::quat quater = glm::quat(t->m_fRotationAngle, glm::normalize(glm::vec3(rotation_angle)));

	//Construct the transformation matrix with glm
	glm::mat4 RotationMat = glm::mat4_cast(glm::normalize(quater));
	glm::mat4 mCTransfor =	glm::translate(glm::mat4(), glm::vec3( t->m_fTranslationx , t->m_fTranslationy, t->m_fTranslationz)) * 
							RotationMat * 
							glm::scale(glm::mat4(), glm::vec3(t->m_fScalar )) * 
							glm::mat4();

	h_inter = 0;

	//Copy the information of the transform an the global itersection boolean
	checkCudaErrors(cudaMemcpy(d_x, glm::value_ptr(mCTransfor),  16 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_inter, &h_inter,  sizeof(unsigned int), cudaMemcpyHostToDevice));

	//Each thread for each triangle
	dim3 BlockDim(threadsxblock, 1, 1); //128 threads per block
	dim3 GridDim(block, 1, 1); 

	//First test with timer
	timer.Start();
	Intercept<<< GridDim, BlockDim, sizeC * sizeof(float3) + sizeof(float3) + sizeof(bool) + 16 *sizeof(float) >>>(d_p1, d_p2, d_A, d_B, sizeC, sizeA, sizeB, d_x, d_inter, d_triID, d_originID, d_destID);
	timer.Stop();

	time += timer.Elapsed();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); //Check for errors

	checkCudaErrors(cudaMemcpy(&h_inter, d_inter,  sizeof(unsigned int), cudaMemcpyDeviceToHost)); //Copy the GPU global intersection variable to know if there was an intersection
	
	if(h_inter != 0) //if there has no been an intersection then do the secon test
	{
		//One thread for each point in the surface
		dim3 BlockDim2(128, 1, 1); //128 threads per block
		dim3 GridDim2((sizeA + BlockDim.x)/BlockDim.x, 1, 1);

		//Second test with timer
		timer.Start();
		SecondTest<<< GridDim2, BlockDim2>>>(d_A, sizeA, d_inter);
		timer.Stop();

		time += timer.Elapsed();
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());//Check for errors

		checkCudaErrors(cudaMemcpy(&h_inter, d_inter,  sizeof(unsigned int ), cudaMemcpyDeviceToHost));  //Copy the GPU global intersection variable to know if there was an intersection	
	}

	//Information of the intersection
	if(h_inter != 0)
	{
		checkCudaErrors(cudaMemcpy(&t->triID, d_triID,  sizeof(unsigned int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&t->originID, d_originID,  sizeof(unsigned int), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(&t->destID, d_destID,  sizeof(unsigned int), cudaMemcpyDeviceToHost));
	}
	else
	{
		t->triID = t->originID = t->destID = 0;
	}


	return h_inter != 0;
}

//Function to copy all the geometry information into the GPU
__host__ void CUDA::Init(float3 * A, uint3 * B, float4 * Normal, float3 * C, unsigned int sA, unsigned int sB, unsigned int sN, unsigned int sC){

	/*for(unsigned int i=0;i<sN;++i)
	{
		printf("%f %f %f \n", Normal[i].x, Normal[i].y, Normal[i].z);
	}*/

	//Check if object C can be stored in shared memory
	cudaDeviceProp prop;

	checkCudaErrors(cudaSetDevice(0));

	checkCudaErrors( cudaGetDeviceProperties( &prop, 0 ) );

	if(prop.sharedMemPerBlock < sC * sizeof(float3) + sizeof(float3) + sizeof(bool) + 16 *sizeof(float))
	{
		printf("Surface C cannot be stored in shared memory. Other approach should be use\n");
		exit(0);
	}

	sizeA = sA;
	sizeB = sB;
	sizeC = sC;
	sizeN = sN;


	threadsxblock = 128;
	block = (sizeB + threadsxblock)/threadsxblock;
	threads = threadsxblock * block;
	

	/* initialize random seed: */
	srand (unsigned int(time(NULL)));

	

	//Allocate memory on the GPU
	checkCudaErrors(cudaMalloc((void**)&d_p1, sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_p2, sizeC * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_A, sizeA * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_B, sizeB * sizeof(uint3)));
	checkCudaErrors(cudaMalloc((void**)&d_Normal, sizeN * sizeof(float4)));
	checkCudaErrors(cudaMalloc((void**)&d_x, 16 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_inter, sizeof(unsigned int )));
	checkCudaErrors(cudaMalloc((void**)&d_triID, sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_originID, sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_destID, sizeof(unsigned int)));
	
	//Send information to the GPU
	checkCudaErrors(cudaMemcpy(d_p1, C, sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_p2, C, sizeC * sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_A, A, sizeA * sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, B, sizeB * sizeof(uint3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Normal, Normal, sizeN * sizeof(float4), cudaMemcpyHostToDevice));


	//Move the point p1 to center
	m_transX = -C[0].x;
	m_transY = -C[0].y;
	m_transZ = -C[0].z;
}

//Function to free memory
__host__ void CUDA::Destroy(){
	//Free memory
	checkCudaErrors(cudaFree(d_p1));
	checkCudaErrors(cudaFree(d_p2));
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_Normal));
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(d_inter));
	checkCudaErrors(cudaFree(d_triID));
	checkCudaErrors(cudaFree(d_originID));
	checkCudaErrors(cudaFree(d_destID));
}