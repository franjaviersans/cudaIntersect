#include "kernel.cuh"


__device__ float distComp(const float3 p1, const float3 p2)
{
	float x, y, z;
	x = (p2.x - p1.x);
	y = (p2.y - p1.y);
	z = (p2.z - p1.z);
	return x * x + y * y + z * z;
}

__device__ float dist(const float3 p1, const float3 p2)
{
	float x, y, z;
	x = (p2.x - p1.x);
	y = (p2.y - p1.y);
	z = (p2.z - p1.z);
	return sqrt(x * x + y * y + z * z);
}

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

__global__ void Intercept(const float3 * const p1, const float3 * const p2,
			   const float3 * const A, const uint3 * const B, const float4 * const normal,
			   const unsigned int sizeC, const unsigned int sizeA,
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
	float4 n;
	float res;
	bool inter = false;
	uint3 id;
	
	if(globalTid < sizeB && blockIdx.y < N) //Each thread works with one triangle in the surface (A, B)
	{

		n = normal[globalTid]; //n have the plane equation of the triangle
		id = B[globalTid]; //Store the points of the triangles in local memory
		
		v0 = A[id.x]; //Point 0
		v1 = A[id.y]; //Point 1
		v2 = A[id.z]; //Point 2


		if(tid == 0) //if it is the first thread of the block
		{
			*origin = *p1; //Copy the data of the origin of the ray
		}

		//Copy all the data of C
		temp = tid;
		while(temp < sizeC)
		{
			//Copy a point of C to local data
			sharedP2[temp] = p2[temp];
			temp += blockDim.x;
		}

		__syncthreads(); //Wait to all the threads in the block


		idN = blockIdx.y;
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
				temp = tid;
				while(temp < sizeC)
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

			//Test 1, check if the center of C is inside the surface of (A,B)
			if(globalinter[idN] == 0) //Only excetue if no intersection have been found
			{
				res = DOT(n, (*originTransformed));

				if(res + n.w > 0.0f) //Check if the point is "in front" of the triangle
				{
					globalinter[idN] = 1;
				}

			}


			//Test 3, check distance with the radious of the sphere
			if(globalinter[idN] == 0) //Only excetue if no intersection have been found
			{
				float d = distComp((*originTransformed), vaux2); //vaux2 contain a point in C. In this case C is a circle and with this formula we calculate the radius
				float d2 = distComp((*originTransformed), v0);

				if(d2 < d) //Check if the point is "in front" of the triangle
				{
					globalinter[idN] = 1;
				}

				d2 = distComp((*originTransformed), v1);

				if(d2 < d) //Check if the point is "in front" of the triangle
				{
					globalinter[idN] = 1;
				}

				d2 = distComp((*originTransformed), v2);

				if(d2 < d) //Check if the point is "in front" of the triangle
				{
					globalinter[idN] = 1;
				}

			}
				
			//Test 3, ray-triangle intersection test, to check if object C is inside (A,B)
			unsigned int i;
			//Only execute if no intersection have been found
			for(i = 0; i < sizeC && globalinter[idN] == 0; ++i)  //For all the points in C do the intersection test
			{
				inter = ray_triangle(v0, v1, v2, (*originTransformed), dir[i]); //Intersection function with the 3 points of the triangle, the origin, and the ith direction
				if(inter) globalinter[idN] = 1;
			}	

			idN += gridDim.y;
		}
	}
}

bool CUDA::CudaIntercept(float &time, float *out_scalar, unsigned int * out_inter, unsigned int N, Transformation &t){
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
	memset(h_inter, 0, N * sizeof(unsigned int));
	

	//Copy the information of the transform an the global itersection boolean
	checkCudaErrors(cudaMemcpy(d_x, h_x, sizeof(mat44) * N, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_inter, h_inter,  sizeof(unsigned int) * N, cudaMemcpyHostToDevice));

	//Each thread for each triangle
	dim3 BlockDim(threadsxblock, 1, 1); //128 threads per block
	dim3 GridDim(block, 2, 1); 

	//First test with timer
	timer.Start();
	Intercept<<< GridDim, BlockDim, sizeC * sizeof(float3) * 2 + sizeof(float3) * 2 + sizeof(mat44) >>>(d_p1, d_p2, d_A, d_B, d_Normal, sizeC, sizeA, sizeB, d_x, N, d_inter);
	timer.Stop();


	time += timer.Elapsed();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); //Check for errors

	checkCudaErrors(cudaMemcpy(h_inter, d_inter,  sizeof(unsigned int) * N, cudaMemcpyDeviceToHost)); //Copy the GPU global intersection variable to know if there was an intersection
	
	memcpy(out_inter, h_inter, sizeof(unsigned int) * N); //Store the intersection 

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
	checkCudaErrors(cudaMalloc((void**)&d_inter, sizeof(unsigned int ) * MAX_N));
	checkCudaErrors(cudaMalloc((void**)&d_x, sizeof(mat44) * MAX_N));
	
	h_x = new mat44 [MAX_N];
	h_inter = new unsigned int [MAX_N];

	float3 h_p1;
	h_p1.x = h_p1.y = h_p1.z = 0.0f;
	
	//Send information to the GPU
	checkCudaErrors(cudaMemcpy(d_p1, (float3 * )&h_p1, sizeof(float3), cudaMemcpyHostToDevice));
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
	delete [] h_x;
	delete [] h_inter;
}