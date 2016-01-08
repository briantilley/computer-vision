#version 140 // GLSL v1.4

uniform usampler2D texImage;

void main(void)
{
	vec4 color = texture(texImage, gl_TexCoord[0].xy);
	gl_FragColor = color / 255.f;
}