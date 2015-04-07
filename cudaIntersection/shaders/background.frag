#version 440

in vec3 vVertexColor;

layout(location = 0) out vec4 vFragColor;


void main(void)
{
	vFragColor = vec4(vVertexColor, 1.0f);
}