
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "kernel.cuh"

#include "rply.h"
#include <stdlib.h>
#include <string>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>

using namespace std;
// use this namespace in this way:
// if (Surface::loadFile(filename)){//code}
// the vectors are vertex and faces
namespace Surface
{
	struct vert
	{
		float x;
		float y;
		float z;
	};
	struct face
	{
		unsigned int id0;
		unsigned int id1;
		unsigned int id2;
	};

	//only use this 2 vectors!
	vector<vert> vertex;
	vector<face> faces;
	
	//temp data to perform the algorithm
	vert tempVertex;
	face tempFace;
	int iIndex = 0;
	
	//vertex callback
	static int vertex_cb(p_ply_argument argument) {
		long eol;
		ply_get_argument_user_data(argument, NULL, &eol);
		switch (iIndex)
		{
		case 0:
			Surface::tempVertex.x = float(ply_get_argument_value(argument));
			iIndex++;
			break;
		case 1:
			Surface::tempVertex.y = float(ply_get_argument_value(argument));
			iIndex++;
			break;
		case 2:
			Surface::tempVertex.z = float(ply_get_argument_value(argument));
			Surface::vertex.push_back(Surface::tempVertex);
			iIndex = 0;
			break;
		}
		return 1;
	}

	//face callback
	static int face_cb(p_ply_argument argument) {
		long length, value_index;
		ply_get_argument_property(argument, NULL, &length, &value_index);
		switch (value_index) {
		case 0:
			tempFace.id0 = (int)ply_get_argument_value(argument);
			break;
		case 1:
			tempFace.id1 = (int)ply_get_argument_value(argument);
			break;
		case 2:
			tempFace.id2 = (int)ply_get_argument_value(argument);
			faces.push_back(tempFace);
			break;
		default:
			break;
		}
		return 1;
	}

	//just call this function
	bool loadFile(string filename)
	{
		long nvertices, ntriangles;
		p_ply ply = ply_open(filename.c_str(), NULL, 0, NULL);
		if (!ply) return false;
		if (!ply_read_header(ply)) return false;
		nvertices= ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);
		ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 0);
		ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 1);
		ntriangles = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, NULL, 0);
		//printf("%ld\n%ld\n", nvertices, ntriangles);
		if (!ply_read(ply)) return false;
		ply_close(ply);
		return true;
	}
}

namespace Sphere
{
	struct vert
	{
		float x;
		float y;
		float z;
	};
	struct face
	{
		unsigned int id0;
		unsigned int id1;
		unsigned int id2;
	};

	//only use this 2 vectors!
	vector<vert> vertex;
	vector<face> faces;
	
	//temp data to perform the algorithm
	vert tempVertex;
	face tempFace;
	int iIndex = 0;
	
	//vertex callback
	static int vertex_cb(p_ply_argument argument) {
		long eol;
		ply_get_argument_user_data(argument, NULL, &eol);
		switch (iIndex)
		{
		case 0:
			tempVertex.x = float(ply_get_argument_value(argument));
			iIndex++;
			break;
		case 1:
			tempVertex.y = float(ply_get_argument_value(argument));
			iIndex++;
			break;
		case 2:
			tempVertex.z = float(ply_get_argument_value(argument));
			vertex.push_back(tempVertex);
			iIndex = 0;
			break;
		}
		return 1;
	}

	//face callback
	static int face_cb(p_ply_argument argument) {
		long length, value_index;
		ply_get_argument_property(argument, NULL, &length, &value_index);
		switch (value_index) {
		case 0:
			tempFace.id0 = (int)ply_get_argument_value(argument);
			break;
		case 1:
			tempFace.id1 = (int)ply_get_argument_value(argument);
			break;
		case 2:
			tempFace.id2 = (int)ply_get_argument_value(argument);
			faces.push_back(tempFace);
			break;
		default:
			break;
		}
		return 1;
	}

	//just call this function
	bool loadFile(string filename)
	{
		long nvertices, ntriangles;
		p_ply ply = ply_open(filename.c_str(), NULL, 0, NULL);
		if (!ply) return false;
		if (!ply_read_header(ply)) return false;
		nvertices= ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);
		ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 0);
		ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 1);
		ntriangles = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, NULL, 0);
		//printf("%ld\n%ld\n", nvertices, ntriangles);
		if (!ply_read(ply)) return false;
		ply_close(ply);
		return true;
	}
}
int main(){
	float total_time = 0;


	if (!Surface::loadFile("surfaceAB.ply"))
		cout << "not loaded" << endl;
	else{
		if (!Sphere::loadFile("sphere.ply"))
			cout << "not loaded" << endl;
		else{
			CUDA c;
			c.Init((float *)(Surface::vertex.data()), 
									(unsigned int* )(Surface::faces.data()), 
									(float *)(Sphere::vertex.data()),
									Surface::vertex.size(), 
									Surface::faces.size(),
									Sphere::vertex.size());
			for(int i = 0;i < M;++i){
				c.CudaIntercept(total_time);
			}

			printf("Total time: %f msecs.\n", total_time);
			printf("Average: %f msecs.\n", total_time / 2000);

			c.Destroy();
		}
	}


	return 0;
}
