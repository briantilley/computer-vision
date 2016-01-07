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

// development
#include <iostream>

// #define ALLOW_WINDOW_RESIZING // makes windows resizeable

class CudaGLviewer
{
private:

	// NEEDED MEMBER VARS (? = maybe unnecessary)
	// unsigned window dimensions
	// unsigned image dimensions
	// GLFWwindow* GLFW window handle
	// GLuint texture
	// cudaGraphicsResource cuda texture handle
	// GLuint fbo (?)
	// cudaGraphicsResource cuda texture screen resource (?)
	// unsigned texture byte size
	// unsigned texture dimensions
	// GLuint texture screen (render target?)
	// GLuint cuda texture (cuda output copy?)

	unsigned m_windowWidth, m_windowHeight;
	unsigned m_imageWidth, m_imageHeight;
	GLFWwindow* m_GLFWwindow;
	// GLuint

	// handle creation/operational failures with grace
	m_isValid = false;

	// CALLBACKS: static methods because GLFW employs a C-style API
	
	// called by GLFW when errors occur
	static cb_GLFWerror(int err, char[] description)
	{
		std::cerr << "GLFW error " << err << ": " << description << std::endl;
	}

	// window closing
	static cb_GLFWcloseWindow(GLFWwindow*);

	// keypresses (revisit this when display works)
	// static cb_GLFWkeyEvent(GLFWwindow*, int key, int scancode, int action, int modifiers);

	// frame buffer size
	static cb_GLFWframebufferSize(GLFWwindow*, int width, int height);

public:

	CudaGLviewer();
	~CudaGLviewer();

	// accessors

	// indicate healthy instance
	operator bool() const { return m_isValid; } // implicit conversion
	bool good() const { return m_isValid; }

};

#endif