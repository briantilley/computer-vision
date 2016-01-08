#include "headers/CudaGLviewer.h"

// load file text
#include <fstream>

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
// glfwWaitEvents must be on a separate thread from glfwSwapBuffers

// OpenGL error handling
#define glErr() glError(glGetError(), __FILE__, __LINE__)
inline void glError(GLenum err, const char file[], uint32_t line, bool abort=true)
{
    if(GL_NO_ERROR != err)
    {
        std::cerr << "[" << file << ":" << line << "] ";
        std::cerr << glewGetErrorString(err) << std::endl;
        if(abort) exit(err);
    }
}

// global state management
bool CudaGLviewer::s_globalStateInitialized = false;
GLuint CudaGLviewer::s_shaderProgram = 0;

// call before instantiating this class
// returns 0 on success, 1 on failure
int CudaGLviewer::initGlobalState()
{
	// call only on first instance creation
	if(GL_FALSE == glfwInit())
	{
		// exit if we can't create any viewers properly
		std::cerr << "CudaGLviewer: error initializing GLFW" << std::endl;
		s_globalStateInitialized = false;
		return 1;
	}

	// call only on first instance creation
	if(NULL == glfwSetErrorCallback(cb_GLFWerror))
	{
		// exit if we can't create any viewers properly
		std::cerr << "CudaGLviewer: error setting GLFW error callback" << std::endl;
		s_globalStateInitialized = false;
		return 1;
	}

	// initialize GLEW
	glewExperimental = GL_TRUE; // use current functionality
	glewInit();
	// expected to spit out "Unknown Error", so don't abort here
	glError(glGetError(), __FILE__, __LINE__, false);

	// create the shader program to use for all windows
	s_shaderProgram = compileShaders(VERTEX_SHADER_FILENAME, FRAGMENT_SHADER_FILENAME);
	glErr();

	s_globalStateInitialized = true;
	return 0;
}

// call after finished using this class
// returns 0 on success, 1 on failure
int CudaGLviewer::destroyGlobalState()
{
	// call only on last instance destruction
	glfwTerminate();

	return 0;
}

// returns 0 on failure, anything else on success
GLuint compileShaders(std::string vertexSourceFilename, std::string fragmentSourceFilename)
{
	// references to each element of the shader
	GLuint vertexShader = 0, fragmentShader = 0, shaderProgram = 0;

	// text inside each file, file handles
	std::string vertexSourceCode, fragmentSourceCode;
	std::ifstream vertexSourceFile(vertexSourceFilename);
	std::ifstream fragmentSourceFile(fragmentSourceFilename);

	// temp string for loading from file
	std::string tempStr;

	// compilation checking
	GLint compileSuccess = 0;
	char temp[256] = {0};
	GLint lenLinkInfoLog = 0;
	GLsizei charsInLog = 0;

	// initialize the program, compilation happens later
	shaderProgram = glCreateProgram();

	// load the files

	if(vertexSourceFile)
		while(!vertexSourceFile.eof())
		{
			std::getline(vertexSourceFile, tempStr, '\n');
			vertexSourceCode += tempStr + '\n';
		}
	else
	{
		std::cerr << "could not open vertex source file" << std::endl;
	}

	if(fragmentSourceFile)
		while(!fragmentSourceFile.eof())
		{
			std::getline(fragmentSourceFile, tempStr, '\n');
			fragmentSourceCode += tempStr + '\n';
		}
	else
	{
		std::cerr << "could not open fragment source file" << std::endl;
	}

	// close source codes
	vertexSourceFile.close(); fragmentSourceFile.close();

	// create individual shader objects
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// bind their source code
	const GLchar* vtx = static_cast<const GLchar*>(vertexSourceCode.c_str());
	const GLchar* frag = static_cast<const GLchar*>(fragmentSourceCode.c_str());
	glShaderSource(vertexShader, 1, &vtx, NULL);
	glShaderSource(fragmentShader, 1, &frag, NULL);

	// compile shaders
	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);

	// check compilation for error status

	// get status from OpenGL
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &compileSuccess);

	if(0 == compileSuccess) // failed to compile
	{
		// print error message
		glGetShaderInfoLog(vertexShader, 256, NULL, temp);
		std::cerr << "vertex shader failed to compile" << std::endl;
		std::cerr << temp << std::endl;

		// destory allocated resources
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
		glDeleteProgram(shaderProgram);
	}
	else
	{
		// bind the shader to its program
		glAttachShader(shaderProgram, vertexShader);
	}

	// get status from OpenGL
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &compileSuccess);

	if(0 == compileSuccess) // failed to compile
	{
		// print error message
		glGetShaderInfoLog(fragmentShader, 256, NULL, temp);
		std::cerr << "fragment shader failed to compile" << std::endl;
		std::cerr << temp << std::endl;

		// destory allocated resources
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
		glDeleteProgram(shaderProgram);
	}
	else
	{
		// bind the shader to its program
		glAttachShader(shaderProgram, fragmentShader);
	}

	// link the shader program together
	glLinkProgram(shaderProgram);

	// check result of shader program compile
	glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &lenLinkInfoLog);

	// print the log if something went wrong
	if(0 < lenLinkInfoLog)
	{
		char* linkLog = static_cast<char*>(malloc(lenLinkInfoLog));
		glGetProgramInfoLog(shaderProgram, lenLinkInfoLog, &charsInLog, linkLog);
		std::cerr << "shader failed to link: " << linkLog << std::endl;
		free(linkLog);
	}

	return shaderProgram;
}

// return 0 on success, 1 on failure
int CudaGLviewer::initGL()
{
	// initialize GLFW

	// set window close callback
	if(NULL == glfwSetWindowCloseCallback(m_GLFWwindow, cb_GLFWcloseWindow))
	{
		s_globalStateInitialized = false;
		return 1;
	}

	// set keypress callback (revisit this when display works)
	// if(NULL == glfwSetKeyCallback(m_GLFWwindow, cb_GLFWkeyEvent))
	// {
	// 	s_globalStateInitialized = false;
	// 	return 1;
	// }

	// set framebuffer size callback
	if(NULL == glfwSetFramebufferSizeCallback(m_GLFWwindow, cb_GLFWframebufferSize))
	{
		s_globalStateInitialized = false;
		return 1;
	}

	// set GLFW window hints
	
	// disable deprecated functionality
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	// window resizeablity
	#ifdef ALLOW_WINDOW_RESIZING
		glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	#else
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	#endif

	// make a window and OpenGL context
	// change 1st NULL to glfwGetPrimaryMonitor() for fullscreen
	m_GLFWwindow = glfwCreateWindow(m_windowWidth, m_windowHeight, m_windowTitle.c_str(), NULL, NULL);

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

	// V-SYNC enabled when not testing for performance
	#ifndef PERFORMANCE_TEST
		glfwSwapInterval(1);
	#else
		glfwSwapInterval(0);
	#endif

	return 0;
}

// return 0 on success, 1 on failure
int CudaGLviewer::initCUDA()
{
	// main compute device will also be used for display
	cudaSetDevice(cudaPrimaryDevice);
}

// allocate the buffers needed for interop
// return 0 on success, 1 on failure
int CudaGLviewer::initBuffers()
{
	// Nvidia: createTextureDst(&m_cudaDestTexture, m_imageWidth, m_imageHeight);
	// make the texure to copy CUDA output frames into
	glGenTextures(1, &m_cudaDestTexture);
	glBindTexture(GL_TEXTURE_2D, m_cudaDestTexture);

	// set its parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// I'm guessing this is a memory allocation
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT, m_imageWidth, m_imageHeight, 0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, NULL);
	glErr();

	// register the texture with CUDA
	cudaErr(cudaGraphicsGLRegisterImage(&m_cudaDestResource, m_cudaDestTexture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

	// map the resource (texure)
	cudaErr(cudaGraphicsMapResources(1, &m_cudaDestResource, 0));

	// bind the texture to a useable cudaArray
	cudaErr(cudaGraphicsSubResourceGetMappedArray(&m_cudaDestArray, m_cudaDestResource, 0, 0));

	// unmap texture (CUDA now has its own handle to the texture)
	cudaErr(cudaGraphicsUnmapResources(1, &m_cudaDestResource, 0));
}

// return 0 on success, 1 on failure
int CudaGLviewer::freeResources()
{
	// destroy the window attached to this instance
	glfwDestroyWindow(m_GLFWwindow);

	// destroy the texture used to display
	glDeleteTextures(1, &m_cudaDestTexture);
	glErr();
	m_cudaDestTexture = 0;

	// no longer a valid window to use, intended behavior
	m_isValid = false;
	return 0;
}

CudaGLviewer::CudaGLviewer()
{
	// initialize OpenGL state
	if(0 != initGL())
		return; // object is set as invalid

	// initialize CUDA state
	if(0 != initCUDA())
		return; // object is set as invalid
}

CudaGLviewer::~CudaGLviewer()
{
	// free all instance-attached resources
	freeResources();
}

// copy data to output texture
int CudaGLviewer::displayFrame(GPUFrame& outputCudaFrame)
{
	// calculate the size of the display texture
	unsigned numTexels = m_imageWidth * m_imageHeight;
	unsigned numColorValues = 4 * numTexels;
	unsigned sizeTexture = sizeof(GLubyte) * numColorValues;

	// copy from output frame to array
	cudaErr(cudaMemcpyToArray(m_cudaDestArray, 0, 0, outputCudaFrame.data(), sizeTexture, cudaMemcpyDeviceToDevice));

	// do some other GL stuff

	// failure
	return 1;
}