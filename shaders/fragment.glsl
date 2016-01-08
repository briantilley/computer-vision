#version 150 // GLSL 1.5

in vec2 Texcoord;

out vec4 outColor;

uniform sampler2D tex;

void main( )
{
	outColor = texture( tex, Texcoord );
}