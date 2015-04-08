#version 440

uniform mat4 mProjection, mModelView;

layout(location = 0) in vec4 vVertex;
layout(location = 2) in vec4 vNormal;

out vec4 vert;
out vec4 norm;

void main()
{
	vert = vVertex;
	norm = vNormal;
	
	gl_Position = mProjection * mModelView * vVertex;
}