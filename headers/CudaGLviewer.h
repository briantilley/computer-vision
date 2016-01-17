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
// #define GLFW_INCLUDE_GLCOREARB
#include <GLFW/glfw3.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// STL
#include <mutex>

// self-rolled
#include "GPUFrame.h"
#include "constants.h"

// development
#include <iostream>

#define ALLOW_WINDOW_RESIZING // makes windows resizeable

#define VERTEX_SHADER_FILENAME "shaders/vertex.glsl"
#define FRAGMENT_SHADER_FILENAME "shaders/fragment.glsl"

#define DEFAULT_WINDOW_WIDTH 640
#define DEFAULT_WINDOW_HEIGHT 360

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
	GLuint m_vertexShader, m_fragmentShader, m_shaderProgram;
	GLuint m_vertexArray, m_vertexBuffer;

	// CUDA/OpenGL interop
	GLuint m_cudaDestTexture; // copy to here from CUDA for display
	struct cudaGraphicsResource* m_cudaDestResource; // resource referring to a GL texture
	cudaArray* m_cudaDestArray; // array that CUDA runtime can use to access texture

	// handle creation/operational failures with grace
	bool m_isValid = false;

	// causing problems
	// // match display intervals to input intervals
	// unsigned m_lastTimestamp, m_lastDisplayTime;

	// macros
	GLuint compileShaders(std::string, std::string);
	int initGL(void);
	int initCUDA(void);
	int initBuffers(void);
	int freeResources(void);

	// called in methods that need to use openGL resources
	std::unique_lock<std::mutex> captureGL(void)
	{
		std::unique_lock<std::mutex> mlock(s_GLlock);
		glfwMakeContextCurrent(m_GLFWwindow);
		return mlock;
	}

	// static data members
	static bool s_globalStateInitialized;
	static std::mutex s_GLlock;

	// CALLBACKS: static methods because GLFW employs a C-style API
	
	// called by GLFW when errors occur
	static void cb_GLFWerror(int err, const char* description)
	{
		std::cerr << "GLFW error " << err << ": " << description << std::endl;
	}

	// window closing
	static void cb_GLFWcloseWindow(GLFWwindow* currentWindow)
	{
		// get the instance that owns this window and invalidate it
		CudaGLviewer* pInstance = static_cast<CudaGLviewer*>(glfwGetWindowUserPointer(currentWindow));
		pInstance->m_isValid = false;
	}

	// keypresses (revisit this when display works)
	// static cb_GLFWkeyEvent(GLFWwindow*, int key, int scancode, int action, int modifiers);

	// frame buffer size
	static void cb_GLFWframebufferSize(GLFWwindow* currentWindow, int width, int height)
	{
		// call glViewport on the proper window
		glfwMakeContextCurrent(currentWindow);
		glViewport(0, 0, width, height);
	}

public:

	// must be called before instantiating CudaGLviewer
	static int initGlobalState(void);

	// undefined behavior if instances still exist when this is called
	static int destroyGlobalState(void);

	// must be called frequently from the main loop
	static void update()
	{ glfwPollEvents(); }

	CudaGLviewer(unsigned imageWidth, unsigned imageHeight, std::string windowTitle);
	~CudaGLviewer();

	// accessors

	// indicate healthy instance
	operator bool() const { return m_isValid; } // implicit conversion
	bool good() const { return m_isValid; }

	// utilities

	// make sure output is m_imageWidth * m_imageHeight
	// pixels of normal RGBA data in aligned memory
	int drawFrame(GPUFrame&);

};

#endif