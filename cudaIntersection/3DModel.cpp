#include "include/GL/glew.h"
#include "3DModel.h"
#include <iostream>
#include "Definitions.h"
#include "OGLBasic.h"

//#define BUFFER_OFFSET(i) ((char *)NULL + (i))
std::vector<Vertex> C3DModel::m_vVertex(0);
std::vector<Mesh> C3DModel::m_vMesh(0);
Vertex C3DModel::m_tempVertex = { 0, 0, 0 };
Mesh C3DModel::m_tempMesh = { 0, 0, 0 };
int C3DModel::m_iTempIndex = 0;
///
/// default constructor
///
C3DModel::C3DModel()
{
	m_uVAO = 0;
	m_uVBOIndex = 0;
	m_uVBO = 0;
	m_iNPoints = 0;
	m_iTempIndex = 0;
	m_vMesh.clear();
	m_vVertex.clear();
}

///
/// default destructor
///
C3DModel::~C3DModel()
{
	m_vMesh.clear();
	m_vVertex.clear();
	deleteBuffers();
	//TRACE("model unloaded\n");
}

void C3DModel::deleteBuffers()
{
	if (m_uVAO != 0)	glDeleteBuffers(1, &m_uVAO);
	if (m_uVBO != 0)	glDeleteBuffers(1, &m_uVBO);
	if (m_uVBOIndex != 0) glDeleteBuffers(1, &m_uVBOIndex);
}

//vertex callback
	int C3DModel::vertex_cb(p_ply_argument argument) {
	long eol;
	ply_get_argument_user_data(argument, NULL, &eol);
	switch (m_iTempIndex)
	{
	case 0:
		m_tempVertex.x = float(ply_get_argument_value(argument));
		m_iTempIndex++;
		break;
	case 1:
		m_tempVertex.y = float(ply_get_argument_value(argument));
		m_iTempIndex++;
		break;
	case 2:
		m_tempVertex.z = float(ply_get_argument_value(argument));
		m_vVertex.push_back(m_tempVertex);
		m_iTempIndex = 0;
		break;
	}
	return 1;
}

//face callback
int C3DModel::face_cb(p_ply_argument argument) {
	long length, value_index;
	ply_get_argument_property(argument, NULL, &length, &value_index);
	switch (value_index) {
	case 0:
		m_tempMesh.id0 = (int)ply_get_argument_value(argument);
		break;
	case 1:
		m_tempMesh.id1 = (int)ply_get_argument_value(argument);
		break;
	case 2:
		m_tempMesh.id2 = (int)ply_get_argument_value(argument);
		m_vMesh.push_back(m_tempMesh);
		break;
	default:
		break;
	}
	return 1;
}


///
/// Function to load a 3D object file
///
/// @param sFilename the filename of the 3d object
///
/// @return true if it is load correctly, false otherwise
///
bool C3DModel::load(const std::string & sFilename)
{
	//TRACE("loading the file %s\n", sFilename.c_str());
	long nvertices;
	p_ply ply = ply_open(sFilename.c_str(), NULL, 0, NULL);
	if (!ply) return false;
	if (!ply_read_header(ply)) return false;
	nvertices = ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);
	ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 0);
	ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 1);
	m_iNTriangles = ply_set_read_cb(ply, "face", "vertex_indices", face_cb, NULL, 0);
	//printf("%ld\n%ld\n", nvertices, ntriangles);
	if (!ply_read(ply)) return false;
	ply_close(ply);

	m_iNPoints = m_vVertex.size();
	m_iNTriangles = m_vMesh.size();

	m_vLocalNormalVertex.resize(m_iNPoints);

	for(int j=0; j < m_iNPoints; ++j) m_vLocalNormalVertex[j].x = 0.0f, m_vLocalNormalVertex[j].y = 0.0f, m_vLocalNormalVertex[j].z = 0.0f;

	Vertex A, B, C, BA, CA;
	PlaneEq t;
	for(unsigned int i=0; i < m_vMesh.size(); ++i)
	{
		A = m_vVertex[m_vMesh[i].id0];
		B = m_vVertex[m_vMesh[i].id1];
		C = m_vVertex[m_vMesh[i].id2];

		SUB(BA, B, A);
		SUB(CA, C, A);

		CROSS(t, BA, CA);
		
		t.w = -(DOT(t, A));

		m_vLocalNormal.push_back(t);

		//Add this normal to every point normal
		m_vLocalNormalVertex[m_vMesh[i].id0].x += m_vLocalNormal[i].x;
		m_vLocalNormalVertex[m_vMesh[i].id0].y += m_vLocalNormal[i].y;
		m_vLocalNormalVertex[m_vMesh[i].id0].z += m_vLocalNormal[i].z;

		m_vLocalNormalVertex[m_vMesh[i].id1].x += m_vLocalNormal[i].x;
		m_vLocalNormalVertex[m_vMesh[i].id1].y += m_vLocalNormal[i].y;
		m_vLocalNormalVertex[m_vMesh[i].id1].z += m_vLocalNormal[i].z;

		m_vLocalNormalVertex[m_vMesh[i].id2].x += m_vLocalNormal[i].x;
		m_vLocalNormalVertex[m_vMesh[i].id2].y += m_vLocalNormal[i].y;
		m_vLocalNormalVertex[m_vMesh[i].id2].z += m_vLocalNormal[i].z;
	}

	//creating the VAO for the model
	glGenVertexArrays(1, &m_uVAO);
	glBindVertexArray(m_uVAO);

		//creating the VBO
		glGenBuffers(1, &m_uVBO);
		glGenBuffers(1, &m_uVBOIndex);

		glBindBuffer(GL_ARRAY_BUFFER, m_uVBO);
		glBufferData(GL_ARRAY_BUFFER, m_iNPoints * sizeof(Vertex) + m_iNPoints * sizeof(Vertex), NULL, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, m_iNPoints * sizeof(Vertex), &m_vVertex[0]);
		glBufferSubData(GL_ARRAY_BUFFER, m_iNPoints * sizeof(Vertex), m_iNPoints * sizeof(Vertex), &m_vLocalNormalVertex[0]);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_uVBOIndex);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_iNTriangles * sizeof(Mesh), &m_vMesh[0], GL_STATIC_DRAW);

		glVertexAttribPointer(WORLD_COORD_LOCATION, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), BUFFER_OFFSET(0)); //Vertex
		glVertexAttribPointer(NORMAL_COORD_LOCATION, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), BUFFER_OFFSET(m_iNPoints * sizeof(Vertex))); //Normals
		glEnableVertexAttribArray(WORLD_COORD_LOCATION);
		glEnableVertexAttribArray(NORMAL_COORD_LOCATION);

	glBindVertexArray(0);	//VAO

	m_vLocalVertex.resize(m_vVertex.size());
	copy(m_vVertex.begin(), m_vVertex.end(), m_vLocalVertex.begin());

	m_vLocalMesh.resize(m_vMesh.size());
	copy(m_vMesh.begin(), m_vMesh.end(), m_vLocalMesh.begin());

	m_vVertex.clear();
	m_vMesh.clear();
	return true;
}

///
/// Method to draw the object
///
void C3DModel::drawObject()
{
	glBindVertexArray(m_uVAO);
	glDrawElements(GL_TRIANGLES, m_iNTriangles * 3, GL_UNSIGNED_INT, BUFFER_OFFSET(0));
	glBindVertexArray(0);
}


///
/// Method to draw only a triangle of the object
///
void C3DModel::drawTriangleObject(unsigned int id)
{
	glBindVertexArray(m_uVAO);
	glDrawRangeElements(GL_TRIANGLES, 0, m_iNTriangles * 3, 60, GL_UNSIGNED_INT, &m_vLocalMesh[0]+id * 3);
	glBindVertexArray(0);
}