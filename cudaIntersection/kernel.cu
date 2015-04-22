#include "kernel.cuh"



texture<float> tC;
texture<float> tAx;
texture<float> tAy;
texture<float> tAz;


//Ray-Triangle function to calculate the intersection
__device__ bool ray_triangle( const float3 V1,  // Triangle vertices
                           const float3 V2,
                           const float3 V3,
                           const float3 O,  //Ray origin
                           const float3 D  //Ray direction
						   )
{
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

__global__ void checkCenter(	const float3 * const p1, 
						const float4 * const normal,
						const unsigned int sizeB,
						const unsigned int N,
						const mat44 * const x,
						unsigned int * globalinter)
{
	//Shared memory declaration
	extern __shared__ char buffer[];

	//All point of C will be stored in the array dir in shared memory
	float3 * origin = (float3 *)&buffer[0]; //Normals in shared memory
	float3 * originTransformed = (float3 *)&buffer[sizeof(float3)]; //Normals in shared memory

	//Id of the thread within a block and within the grid
	unsigned int tid = threadIdx.x;
	unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int idN;

	//Auxiliar variables
	if(globalTid < sizeB && blockIdx.z < N) //Each thread works with one triangle in the surface (A, B)
	{

		float4 n = normal[globalTid]; //n have the plane equation of the triangle



		if(tid == 0) //if it is the first thread of the block
		{
			*origin = *p1; //Copy the data of the origin of the ray
		}
		__syncthreads(); //Wait to all the threads in the block


		idN = blockIdx.z;
		while (idN < N){
			if(globalinter[idN] == 0) //Only check if no intersection have been found
			{ 
				if(tid == 0) //if it is the first thread of the block
				{
					MULT((*originTransformed), x[idN].data, (*origin)); //Transfor the point. This is the only transformation done with global transformation data!!!
				}
			}
			__syncthreads(); //Wait to all the threads in the block

			//Test 1, check if the center of C is inside the surface of (A,B)
			if(globalinter[idN] == 0) //Only excetue if no intersection have been found
			{
				if(DOT(n, (*originTransformed)) + n.w > 0.0f) //Check if the point is "in front" of the triangle
				{
					globalinter[idN] = 1;
				}
			}

			idN += gridDim.z;
		}
	}

}

__global__ void Intercept(const float3 * const p1,
			   const unsigned int sizeC,
			   const unsigned int sizeB, 
			   const mat44 * const x,
			   const unsigned int N,
			   unsigned int * globalinter)
{

	//Shared memory declaration
	extern __shared__ char buffer[];

	//All point of C will be stored in the array dir in shared memory
	float3 * sharedP2 = (float3 *)&buffer[0];
	float3 * dir = (float3 *)&buffer[sizeC * sizeof(float3)];
    float3 * origin = (float3 *)&buffer[sizeC * sizeof(float3) * 2]; //The origin will be shared to
	float3 * originTransformed = (float3 *)&buffer[sizeC * sizeof(float3) * 2 + sizeof(float3)]; //The origin will be shared to
	float * lt = (float *)&buffer[sizeC * sizeof(float3) * 2 + sizeof(float3) * 2]; //The shared transformation matrix

	//Id of the thread within a block and within the grid
	unsigned int tid = threadIdx.x;
	unsigned int globalTid = blockDim.x * blockIdx.x + threadIdx.x, temp;
	unsigned int idN;

	//Auxiliar variables
	float3 v0, v1, v2, vaux1, vaux2;
	
	if(globalTid < sizeB && blockIdx.z < N) //Each thread works with one triangle in the surface (A, B)
	{
		v0.x = tex1Dfetch(tAx, globalTid); //Point 0
		v1.x = tex1Dfetch(tAx, globalTid + sizeB);//Point 1
		v2.x = tex1Dfetch(tAx, globalTid + sizeB * 2);//Point 2

		v0.y = tex1Dfetch(tAy, globalTid); //Point 0
		v1.y = tex1Dfetch(tAy, globalTid + sizeB);//Point 1
		v2.y = tex1Dfetch(tAy, globalTid + sizeB * 2);//Point 2

		v0.z = tex1Dfetch(tAz, globalTid); //Point 0
		v1.z = tex1Dfetch(tAz, globalTid + sizeB);//Point 1
		v2.z = tex1Dfetch(tAz, globalTid + sizeB * 2);//Point 2
		


		if(tid == 0) //if it is the first thread of the block
		{
			*origin = *p1; //Copy the data of the origin of the ray
		}

		//Copy all the data of C
		temp = tid;
		while(temp < sizeC)
		{
			//Copy a point of C to local data
			sharedP2[temp].x = tex1Dfetch(tC, temp * 3);
			sharedP2[temp].y = tex1Dfetch(tC, temp * 3 + 1);
			sharedP2[temp].z = tex1Dfetch(tC, temp * 3 + 2);
			temp += blockDim.x;
		}

		__syncthreads(); //Wait to all the threads in the block


		idN = blockIdx.z;
		while (idN < N){
			if(globalinter[idN] == 0) //Only check if no intersection have been found
			{ 
				if(tid == 0) //if it is the first thread of the block
				{
					MULT((*originTransformed), x[idN].data, (*origin)); //Transfor the point. This is the only transformation done with global transformation data!!!
				}

				temp = tid;
				while(temp < 16) //16 values of the 4x4 transformation matrix
				{
					lt[temp] = x[idN].data[temp]; 
					temp += blockDim.x;
					
				}
			}
			__syncthreads(); //Wait to all the threads in the block

			if(globalinter[idN] == 0) //Only check if no intersection have been found
			{ 
				//Transform all the points in C
				temp = tid + blockIdx.y * (sizeC / gridDim.y);
				while(temp < (blockIdx.y + 1) * (sizeC / gridDim.y))
				{
					//Copy a point of C to local data
					vaux1 = sharedP2[temp];

					//Transform the point 
					MULT(vaux2, lt, vaux1);

					//store the direction of the ray in shared memory x(p2) - x(p1)
					dir[temp].x = vaux2.x - (*originTransformed).x;
					dir[temp].y = vaux2.y - (*originTransformed).y;
					dir[temp].z = vaux2.z - (*originTransformed).z;

					temp += blockDim.x;
				}
			}
			__syncthreads(); //Wait to all the threads in the block
				
			//Test 2, ray-triangle intersection test, to check if object C is inside (A,B)
			unsigned int i;
			//Only execute if no intersection have been found
			for(i = blockIdx.y * (sizeC / gridDim.y); i < sizeC && i < (blockIdx.y + 1) * (sizeC / gridDim.y) && globalinter[idN] == 0; ++i)  //For all the points in C do the intersection test
			{
				if(ray_triangle(v0, v1, v2, (*originTransformed), dir[i])) globalinter[idN] = 1;
				//Intersection function with the 3 points of the triangle, the origin, and the ith direction
			}	

			idN += gridDim.z;
		}
	}
}

bool CUDA::CudaIntercept(float &time, float *out_scalar, unsigned int * out_inter, unsigned int N, Transformation &t, unsigned int gridX, unsigned int gridY, unsigned int gridZ, unsigned int blockX){
	GpuTimer timer;

	Transformation * T = new Transformation[N];
	
	//Generate N random transformations
	glm::quat quater;
	glm::vec3 rotation_angle;
	glm::mat4 RotationMat;
	glm::mat4 mCTransfor;
	for(unsigned int i = 0; i < N; ++ i)
	{
		//Generate a random transform with scaling, translating and rotating

		T[i].m_fTranslationx = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f + m_transX;
		T[i].m_fTranslationy = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f + m_transY;
		T[i].m_fTranslationz = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f + m_transZ;

		T[i].m_fScalar = (rand() % RAND_MAX) / float(RAND_MAX * 2.0f) + 0.5f;
	
		rotation_angle = glm::normalize(glm::vec3((rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f, (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f, (rand() % RAND_MAX) / float(RAND_MAX/2.0f) -1.0f));
		T[i].m_fRotationAngle = (rand() % RAND_MAX) / float(RAND_MAX/2.0f) - 1.0f;
		T[i].m_fRotationVectorx = rotation_angle.x;
		T[i].m_fRotationVectory = rotation_angle.y;
		T[i].m_fRotationVectorz = rotation_angle.z;

		//Generate quaternion
		 quater = glm::quat(T[i].m_fRotationAngle, glm::normalize(glm::vec3(rotation_angle)));

		//Construct the transformation matrix with glm
		RotationMat = glm::mat4_cast(glm::normalize(quater));
		mCTransfor  = glm::translate(glm::mat4(), glm::vec3( T[i].m_fTranslationx , T[i].m_fTranslationy, T[i].m_fTranslationz)) * 
								RotationMat * 
								glm::scale(glm::mat4(), glm::vec3(T[i].m_fScalar )) * 
								glm::mat4();

		//Store to pass to the GPU
		memcpy(&h_x[i], glm::value_ptr(mCTransfor), sizeof(mat44));
		out_scalar[i] = T[i].m_fScalar; //Store only the scaling
	}

	//Set the intersection to false
	memset(out_inter, 0, N * sizeof(unsigned int));
	

	//Copy the information of the transform an the global itersection boolean
	checkCudaErrors(cudaMemcpy(d_x, h_x, sizeof(mat44) * N, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_inter, out_inter,  sizeof(unsigned int) * N, cudaMemcpyHostToDevice));

	//Each thread for each triangle
	checkCenter<<< dim3(gridX, 1, 2), dim3(blockX, 1, 1), sizeof(float3) * 2>>>(d_p1, d_Normal, sizeB, N, d_x, d_inter);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); //Check for errors;

	//First test with timer
	timer.Start();
	Intercept<<< dim3(gridX, gridY, gridZ), dim3(blockX, 1, 1), sizeC * sizeof(float3) * 2 + sizeof(float3) * 2 + sizeof(mat44) >>>(d_p1, sizeC, sizeB, d_x, N, d_inter);
	timer.Stop();


	time += timer.Elapsed();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); //Check for errors

	checkCudaErrors(cudaMemcpy(out_inter, d_inter,  sizeof(unsigned int) * N, cudaMemcpyDeviceToHost)); //Copy the GPU global intersection variable to know if there was an intersection

	bool solution = false;

	t.m_fScalar = -1.0f;

	//Find the best intersection for the iteration
	for(unsigned int i=0; i < N; ++i)
	{
		if(out_inter[i] == 0 && T[i].m_fScalar > t.m_fScalar)
		{
			t = T[i];
			solution = true;
		}
	}

	if(!solution) t = T[0];

	delete [] T;

	return solution;
}

//Function to copy all the geometry information into the GPU
__host__ void CUDA::InitOld(float3 * A, uint3 * B, float4 * Normal, float3 * C, unsigned int sA, unsigned int sB, unsigned int sN, unsigned int sC){

	//Check if object C can be stored in shared memory
	/*cudaDeviceProp prop;

	checkCudaErrors(cudaSetDevice(0));

	checkCudaErrors( cudaGetDeviceProperties( &prop, 0 ) );

	if(prop.sharedMemPerBlock < sC * sizeof(float3) * 2 + sizeof(float3) * 2 + sizeof(bool) + 16 *sizeof(float))
	{
		printf("Surface C cannot be stored in shared memory. Other approach should be use\n");
		exit(0);
	}

	

	sizeA = sA;
	sizeB = sB;
	sizeC = sC;
	sizeN = sN;
	

	// initialize random seed: 
	srand (unsigned int(time(NULL)));

	

	//Allocate memory on the GPU
	checkCudaErrors(cudaMalloc((void**)&d_p1, sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_p2, sizeC * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_Ax, sizeA * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Ay, sizeA * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Az, sizeA * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_B, sizeB * sizeof(uint3)));
	checkCudaErrors(cudaMalloc((void**)&d_Normal, sizeN * sizeof(float4)));
	checkCudaErrors(cudaMalloc((void**)&d_inter, sizeof(unsigned int ) * MAX_N));
	checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(mat44) * MAX_N));
	
	h_x = new mat44 [MAX_N];

	
	//Send information to the GPU
	checkCudaErrors(cudaMemcpy(d_p1, C, sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_p2, C, sizeC * sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_A, A, sizeA * sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, B, sizeB * sizeof(uint3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Normal, Normal, sizeN * sizeof(float4), cudaMemcpyHostToDevice));


	//Move the point p1 to center
	m_transX = -C[0].x;
	m_transY = -C[0].y;
	m_transZ = -C[0].z;*/
}

//Function to copy all the geometry information into the GPU
__host__ void CUDA::Init(float3 * A, uint3 * B, float4 * Normal, float3 * C, unsigned int sA, unsigned int sB, unsigned int sN, unsigned int sC){

	//Check if object C can be stored in shared memory
	cudaDeviceProp prop;

	checkCudaErrors(cudaSetDevice(0));

	checkCudaErrors( cudaGetDeviceProperties( &prop, 0 ) );

	if(prop.sharedMemPerBlock < sC * sizeof(float3) * 2 + sizeof(float3) * 2 + sizeof(bool) + 16 *sizeof(float))
	{
		printf("Surface C cannot be stored in shared memory. Other approach should be use\n");
		exit(0);
	}

	//Copy the data to reorder and duplicate points
	
	float * newA = new float[sB * 3 * 3];

	for(unsigned int i = 0, j = 0; i < sB; ++i)
	{
		uint3 id = B[i]; //Triangle
		
		newA[j] = A[id.x].x; //Point 0
		newA[sB + j] = A[id.y].x; //Point 0
		newA[sB * 2 + j] = A[id.z].x; //Point 0

		newA[sB * 3 + j] = A[id.x].y; //Point 1
		newA[sB * 4 + j] = A[id.y].y; //Point 1
		newA[sB * 5 + j] = A[id.z].y; //Point 1

		newA[sB * 6 + j] = A[id.x].z; //Point 2
		newA[sB * 7 + j] = A[id.y].z; //Point 2
		newA[sB * 8 + j] = A[id.z].z; //Point 2
		
		++j;
	}

	sizeA = sB * 3;
	sizeB = sB;
	sizeC = sC - 1;
	sizeN = sN;

	/* initialize random seed: */
	srand (unsigned int(time(NULL)));

	//Allocate memory on the GPU
	checkCudaErrors(cudaMalloc((void**)&d_p1, sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_p2, sizeC * sizeof(float3)));
	checkCudaErrors(cudaMalloc((void**)&d_Ax, sB * 3 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Ay, sB * 3 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_Az, sB * 3 * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_B, sizeB * sizeof(uint3)));
	checkCudaErrors(cudaMalloc((void**)&d_Normal, sizeN * sizeof(float4)));
	checkCudaErrors(cudaMalloc((void**)&d_inter, sizeof(unsigned int ) * MAX_N));
	checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(mat44) * MAX_N));
	h_x = new mat44 [MAX_N];

	//Send information to the GPU
	checkCudaErrors(cudaMemcpy(d_p1, C, sizeof(float3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_p2, (C + 1), sizeC * sizeof(float3), cudaMemcpyHostToDevice));


	cudaBindTexture( NULL, tC, (float * )d_p2, sizeC * sizeof(float3) );

	checkCudaErrors(cudaMemcpy(d_Ax, newA, sB * 3 * sizeof(float) , cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Ay, newA + sB * 3, sB * 3 * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Az, newA + sB * 3 * 2, sB * 3 * sizeof(float), cudaMemcpyHostToDevice));

	cudaBindTexture( NULL, tAx, d_Ax, sB * 3 * sizeof(float) );
	cudaBindTexture( NULL, tAy, d_Ay, sB * 3 * sizeof(float) );
	cudaBindTexture( NULL, tAz, d_Az, sB * 3 * sizeof(float) );

	checkCudaErrors(cudaMemcpy(d_B, B, sizeB * sizeof(uint3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Normal, Normal, sizeN * sizeof(float4), cudaMemcpyHostToDevice));

	delete newA;

	//Move the point p1 to center
	m_transX = -C[0].x;
	m_transY = -C[0].y;
	m_transZ = -C[0].z;
}

//Function to free memory
__host__ void CUDA::Destroy(){

	//Free memory
	checkCudaErrors(cudaFree(d_p1));

	cudaUnbindTexture( tC );

	checkCudaErrors(cudaFree(d_p2));


	cudaUnbindTexture( tAx );
	cudaUnbindTexture( tAy );
	cudaUnbindTexture( tAz );

	checkCudaErrors(cudaFree(d_Ax));
	checkCudaErrors(cudaFree(d_Ay));
	checkCudaErrors(cudaFree(d_Az));

	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_Normal));
	checkCudaErrors(cudaFree(d_x));
	checkCudaErrors(cudaFree(d_inter));
	delete [] h_x;
	cudaDeviceReset();
}