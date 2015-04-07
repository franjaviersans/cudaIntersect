#version 440

uniform mat4 mProjection;

layout(location = 0) in vec4 vVertex;
layout(location = 3) in vec3 vColor;

out vec3 vVertexColor;

void main()
{
	vVertexColor = vColor;
	gl_Position = mProjection * vVertex;
}