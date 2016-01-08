#version 140 // GLSL v1.4

void main(void)
{
	gl_Position = glVertex;
	glTexCoord[0].xy = gl_MultiTexCoord0.xy;
}