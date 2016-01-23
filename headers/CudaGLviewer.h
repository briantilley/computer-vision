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

class GLcontextWrapper
{
public:
	GLcontextWrapper() = default;

	void lock(GLFWwindow* context)
	{
		glfwMakeContextCurrent(context);
	}

	~GLcontextWrapper()
	{
		glfwMakeContextCurrent(NULL);
	}
};

class CudaGLviewer
{
private:

	// OpenGL error handling
	#define glErr() glError(glGetError(), __FILE__, __LINE__)
	static inline void glError(GLenum err, const char file[], uint32_t line, bool abort=true)
	{
		if(GL_NO_ERROR != err)
		{
			std::cerr << "[" << file << ":" << line << "] ";
			std::cerr << glewGetErrorString(err) << " " << err << std::endl;
			if(abort) exit(err);
		}
	}

	/*
	 * GLFW multithreading restrictions
	 *
	 * main thread:
	 * 		glfwSetErrorCallback
	 * 		glfwInit
	 * 		glfwTerminate
	 * 		glfwWindowHint
	 *  	glfwCreateWindow
	 * 		glfwSetWindowCloseCallback
	 * 		glfwSetKeyCallback
	 * 		glfwSetFramebufferSizeCallback
	 * 		glfwDestroyWindow
	 *
	 * any thread:
	 * 		glfwSetWindowUserPointer
	 * 		glfwSwapInterval
	 * 		glfwSwapBuffers
	 */

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

	// track when to close the window
	bool m_shouldClose = false;

	// running thread needs to call glViewport to resize window
	bool m_windowResized = false;

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
	std::unique_lock<std::mutex> captureGL(GLcontextWrapper& ctx) const
	{
		cudaErr(cudaSetDevice(cudaPrimaryDevice)); // set CUDA device for good measure
		std::unique_lock<std::mutex> mlock(s_GLlock);
		// glfwMakeContextCurrent(m_GLFWwindow);
		ctx.lock(m_GLFWwindow);
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
		pInstance->m_shouldClose = true;
	}

	// keypresses (revisit this when display works)
	// static cb_GLFWkeyEvent(GLFWwindow*, int key, int scancode, int action, int modifiers);

	// frame buffer size
	static void cb_GLFWframebufferSize(GLFWwindow* currentWindow, int width, int height)
	{
		// get attached instance for video info
		CudaGLviewer* pInstance = static_cast<CudaGLviewer*>(glfwGetWindowUserPointer(currentWindow));

		// update dimensions
		pInstance->m_windowWidth = width; pInstance->m_windowHeight = height;

		// take control of OpenGL
		GLcontextWrapper contextWrapper;
		auto lock = pInstance->captureGL(contextWrapper);

		// vertical, horizontal, or no borders
		int deltaCrossMultiplication = width * pInstance->m_imageHeight - height * pInstance->m_imageWidth;
		if(0 < deltaCrossMultiplication) // vertical bars
		{
			GLfloat scaledX = (static_cast<GLfloat>(pInstance->m_imageWidth) * height) / (width * pInstance->m_imageHeight);

			GLfloat scaled[] = {
				//     X     Y    U    V
				-scaledX,  1.f, 0.f, 0.f, // upper left
				 scaledX,  1.f, 1.f, 0.f, // lower right
				-scaledX, -1.f, 0.f, 1.f, // lower left

				-scaledX, -1.f, 0.f, 1.f, // lower left
				 scaledX, -1.f, 1.f, 1.f, // lower right
				 scaledX,  1.f, 1.f, 0.f  // upper right
			};

			// bind buffer, copy new vertices, unbind
			glBindBuffer(GL_ARRAY_BUFFER, pInstance->m_vertexBuffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(scaled), scaled, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
		else if(0 > deltaCrossMultiplication) // horizontal bars
		{
			GLfloat scaledY = (static_cast<GLfloat>(pInstance->m_imageHeight) * width) / (height * pInstance->m_imageWidth);

			GLfloat scaled[] = {
				// X         Y    U    V
				-1.f,  scaledY, 0.f, 0.f, // upper left
				 1.f,  scaledY, 1.f, 0.f, // lower right
				-1.f, -scaledY, 0.f, 1.f, // lower left

				-1.f, -scaledY, 0.f, 1.f, // lower left
				 1.f, -scaledY, 1.f, 1.f, // lower right
				 1.f,  scaledY, 1.f, 0.f  // upper right
			};

			// bind buffer, copy new vertices, unbind
			glBindBuffer(GL_ARRAY_BUFFER, pInstance->m_vertexBuffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(scaled), scaled, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
		else // no bars
		{
			GLfloat scaled[] = {
				// X     Y    U    V
				-1.f,  1.f, 0.f, 0.f, // upper left
				 1.f,  1.f, 1.f, 0.f, // lower right
				-1.f, -1.f, 0.f, 1.f, // lower left

				-1.f, -1.f, 0.f, 1.f, // lower left
				 1.f, -1.f, 1.f, 1.f, // lower right
				 1.f,  1.f, 1.f, 0.f  // upper right
			};
			// bind buffer, copy new vertices, unbind
			glBindBuffer(GL_ARRAY_BUFFER, pInstance->m_vertexBuffer);
			glBufferData(GL_ARRAY_BUFFER, sizeof(scaled), scaled, GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

		std::cout << "resize " << pInstance->m_windowWidth << " " << pInstance->m_windowHeight << std::endl;

		// signal resize event
		// pInstance->m_windowResized = true;
		// // call glViewport to apply changes
		glViewport(0, 0, pInstance->m_windowWidth, pInstance->m_windowHeight);
		glErr();
	}

public:

	// must be called before instantiating CudaGLviewer
	static int initGlobalState(void);

	// undefined behavior if instances still exist when this is called
	static int destroyGlobalState(void);

	// must be called frequently from the main loop
	static void update()
	{ glfwPollEvents(); }

	// instances must be created in the main thread
	CudaGLviewer(unsigned imageWidth, unsigned imageHeight, std::string windowTitle);
	~CudaGLviewer();

	// call this in the running thread
	int initialize(void);

	// accessors

	// indicate healthy instance
	operator bool() const { return m_isValid && !m_shouldClose; } // implicit conversion
	bool good(void) const { return m_isValid && !m_shouldClose; }
	bool shouldClose(void) const { return m_shouldClose; }

	// utilities

	// make sure output is m_imageWidth * m_imageHeight
	// pixels of normal RGBA data in aligned memory
	int drawFrame(GPUFrame&);

};

#endif