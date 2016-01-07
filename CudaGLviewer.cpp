#include "CudaGLviewer.h"

// write for one instance, then adapt for N instances

// NEEDED FUNCTIONS (? = maybe unnecessary)
// init gl
// init cuda
// free resources
// create output texture
// delete output texture
// display callback (?)
// idle callback (?)
// GLFW error callback

// return 0 on success, 1 on failure
int CudaGLviewer::initGL()
{
	// call only on first instance creation
	if(!glfwInit())
	{
		m_isValid = false;
		return 1;
	}

	// call only on first instance creation
	glfwSetErrorCallback(cb_GLFWerror);

	// set GLFW window hints
	#ifdef ALLOW_WINDOW_RESIZING
		glfwWindowHint(GLFW_RESIZEABLE, GL_TRUE);
	#else
		glfwWindowHint(GLFW_RESIZEABLE, GL_FALSE);
	#endif

	// make a window and OpenGL context
	// change 1st NULL to glfwGetPrimaryMonitor() for fullscreen
	m_GLFWwindow = glfwCreateWindow(m_windowWidth, m_windowHeight, m_windowTitle, NULL, NULL);

	// NULL returned if creation fails
	if(NULL == m_GLFWwindow)
	{
		m_isValid = false;
		return 1;
	}

	// set the current context for state-based OpenGL
	glfwMakeContextCurrent(m_GLFWwindow);
}

// return 0 on success, 1 on failure
int CudaGLviewer::initCUDA()
{

}

// return 0 on success, 1 on failure
int CudaGLviewer::freeResources()
{
	// destroy the window attached to this instance
	glfwDestroyWindow(m_GLFWwindow);

	// call only on last instance destruction
	glfwTerminate();

	// no longer a valid window to use, intended behavior
	m_isValid = false;
	return 0;
}

CudaGLviewer::CudaGLviewer()
{

}

CudaGLviewer::~CudaGLviewer()
{

}