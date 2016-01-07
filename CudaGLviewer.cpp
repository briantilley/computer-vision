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

// Note: 'windows' and 'contexts' are inseperable and mutually exclusive,
// so they are referred to interchangeably

// put glfwSwapBuffers and glfwWaitEvents somewhere

// return 0 on success, 1 on failure
int CudaGLviewer::initGL()
{
	// call only on first instance creation
	if(GL_FALSE == glfwInit())
	{
		// exit if we can't create any viewers properly
		std::cerr << "CudaGLviewer: error initializing GLFW" << std::endl;
		exit(EXIT_FAILURE);
	}

	// call only on first instance creation
	if(NULL == glfwSetErrorCallback(cb_GLFWerror))
	{
		// exit if we can't create any viewers properly
		std::cerr << "CudaGLviewer: error setting GLFW error callback" << std::endl;
		exit(EXIT_FAILURE);
	}

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

	// attach this instance to the window to make life easier
	glfwSetWindowUserPointer(m_GLFWwindow, this);

	// set the current context for state-based OpenGL
	glfwMakeContextCurrent(m_GLFWwindow);

	// set window close callback
	// call only on first instance creation
	if(NULL == glfwSetWindowCloseCallback(m_GLFWwindow, cb_GLFWcloseWindow))
	{
		m_isValid = false;
		return 1;
	}

	// set keypress callback (revisit this when display works)
	// glfwSetKeyCallback(m_GLFWwindow, cb_GLFWkeyEvent);

	// get window size
	glfwGetFrameBufferSize(m_GLFWwindow, &m_windowWidth, &m_windowHeight);
	
	// set the OpenGL viewport accordingly
	glViewport(0, 0, m_windowWidth, m_windowHeight);

	// set a callback to do the above action when needed
	// call only on first instance creation
	if(NULL == glfwSetFramebufferSizeCallback(m_GLFWwindow, cb_GLFWframebufferSize))
	{
		m_isValid = false;
		return 1;
	}
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