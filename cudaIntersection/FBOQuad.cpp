// $Id: $
//
// Author: Francisco Sans franjaviersans@gmail.com
//
// Complete history on bottom of file

#define FILE_REVISION "$Revision: $"

#include "FBOQuad.h"
#include <stdlib.h>


// Global static pointer used to ensure a single instance of the class.
FBOQuad * FBOQuad::m_quad = NULL;

/**
* Default constructor
*/
FBOQuad::FBOQuad()
{

	GLfloat Vertex[] = {//Word          Color
						-0.5f,-0.5f,	0.46f,0.46f, 0.92f,
						0.5f,-0.5f,		0.46f,0.46f, 0.92f,
						0.5f,0.5f,		0.08f,0.08f, 0.16f,
						-0.5f,0.5f,		0.08f,0.08f, 0.16f
						}; 


	GLuint Indices[] = {0,1,2,3}; 



	glGenBuffers(1, &m_iVBO);
	glGenBuffers(1, &m_iVBOIndex);

	// bind buffer for positions and copy data into buffer
	// GL_ARRAY_BUFFER is the buffer type we use to feed attributes
	glBindBuffer(GL_ARRAY_BUFFER, m_iVBO);
 
		// feed the buffer, and let OpenGL know that we don't plan to
		// change it (STATIC) and that it will be used for drawing (DRAW)
		glBufferData(GL_ARRAY_BUFFER, 20 * sizeof(GL_FLOAT), Vertex, GL_STATIC_DRAW);

		//bind buffer for positions and copy data into buffer
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_iVBOIndex);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * sizeof(GL_UNSIGNED_INT), Indices, GL_STATIC_DRAW);


	//Generate the VAO
	glGenVertexArrays(1, &m_iVAO);
    glBindVertexArray(m_iVAO );

		// bind buffer for positions and copy data into buffer
		// GL_ARRAY_BUFFER is the buffer type we use to feed attributes
		glBindBuffer(GL_ARRAY_BUFFER, m_iVBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_iVBOIndex);

		glEnableVertexAttribArray(WORLD_COORD_LOCATION);
		glVertexAttribPointer(WORLD_COORD_LOCATION, 2, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 5, BUFFER_OFFSET(0)); //Vertex
		glEnableVertexAttribArray(COLOR_COORD_LOCATION);
		glVertexAttribPointer(COLOR_COORD_LOCATION, 3, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 5, BUFFER_OFFSET(sizeof(GL_FLOAT)*2)); //Text Coord
		
	//Unbind the vertex array	
	glBindVertexArray(0);


	//Disable Buffers and vertex attributes
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

/**
* Creates an instance of the model
*
* @return an instance of this class
*/
FBOQuad * FBOQuad::Instance() 
{
	if (!m_quad)   // Only allow one instance of class to be generated.
		 m_quad = new FBOQuad;
 
   return m_quad;


}

/**
* Default destructor
*/
FBOQuad::~FBOQuad()
{

	glDeleteBuffers(1, &m_iVBO);
	glDeleteBuffers(1, &m_iVBOIndex);
	glDeleteVertexArrays(1, &m_iVAO);
}

/**
* Method to Draw the Quad
*/
void FBOQuad::Draw()
{

	//Bind Buffers
	glBindVertexArray(m_iVAO );

		glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);

	//Unbid Buffer
	glBindVertexArray(0);

}


/**
* Method start up Quad rendering
*/
void FBOQuad::StartUp()
{
	//Bind Buffers
	glBindVertexArray(m_iVAO );
}


/**
* Method to only Draw the Quad
*/
void FBOQuad::OnlyDraw()
{
	glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, 0);
}


/**
* Method to stop Quad rendering
*/
void FBOQuad::Stop(){
	//Unbid Buffer
	glBindVertexArray(0);
}


#undef FILE_REVISION

// Revision History:
// $Log: $
// $Header: $
// $Id: $