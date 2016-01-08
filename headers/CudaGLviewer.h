#ifndef CUDA_GL_VIEWER
#define CUDA_GL_VIEWER

/*
 * Wrapper class for displaying images from
 * CUDA global memory to an opengl window
 * (based on Nvidia's simpleCUDA2GL sample
 * located in 3_Imaging from CUDA toolkit samples)
 */

// OpenGL
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// self-rolled
#include "GPUFrame.h"
#include "constants.h"

// development
#include <iostream>

// #define ALLOW_WINDOW_RESIZING // makes windows resizeable

#define VERTEX_SHADER_FILENAME "shaders/vertex.glsl"
#define FRAGMENT_SHADER_FILENAME "shaders/fragment.glsl"

extern unsigned cudaPrimaryDevice;
extern unsigned cudaSecondaryDevice;

class CudaGLviewer
{
private:

	// NEEDED MEMBER VARS (? = maybe unnecessary)
	// unsigned window dimensions :)
	// unsigned image dimensions :)
	// GLFWwindow* GLFW window handle :)
	// GLuint texture
	// cudaGraphicsResource cuda texture handle :)
	// GLuint fbo (?)
	// cudaGraphicsResource cuda texture screen resource (?)
	// unsigned texture byte size
	// unsigned texture dimensions
	// GLuint texture screen (render target?)
	// GLuint cuda texture (cuda output copy?)

	// OpenGL member variables
	unsigned m_windowWidth, m_windowHeight;
	unsigned m_imageWidth, m_imageHeight;
	GLFWwindow* m_GLFWwindow;
	std::string m_windowTitle;

	// CUDA/OpenGL interop
	GLuint m_cudaDestTexture; // copy to here from CUDA for display
	struct cudaGraphicsResource* m_cudaDestResource; // resource referring to a GL texture
	cudaArray* m_cudaDestArray; // array that CUDA runtime can use to access texture

	// handle creation/operational failures with grace
	bool m_isValid = false;

	// macros
	static GLuint compileShaders(std::string, std::string);
	int initGL(void);
	int initCUDA(void);
	int initBuffers(void);
	int freeResources(void);

	// static data members
	static bool s_globalStateInitialized;
	static GLuint s_shaderProgram;

	// CALLBACKS: static methods because GLFW employs a C-style API
	
	// called by GLFW when errors occur
	static void cb_GLFWerror(int err, const char* description)
	{
		std::cerr << "GLFW error " << err << ": " << description << std::endl;
	}

	// window closing
	static void cb_GLFWcloseWindow(GLFWwindow*);

	// keypresses (revisit this when display works)
	// static cb_GLFWkeyEvent(GLFWwindow*, int key, int scancode, int action, int modifiers);

	// frame buffer size
	static void cb_GLFWframebufferSize(GLFWwindow*, int width, int height);

public:

	// must be called before instantiating CudaGLviewer
	static int initGlobalState(void);

	// undefined behavior if instances still exist when this is called
	static int destroyGlobalState(void);

	CudaGLviewer();
	~CudaGLviewer();

	// accessors

	// indicate healthy instance
	operator bool() const { return m_isValid; } // implicit conversion
	bool good() const { return m_isValid; }

	// utilities

	// make sure output is m_imageWidth * m_imageHeight
	// pixels of normal RGBA data in aligned memory
	int displayFrame(GPUFrame&);

};

#endif