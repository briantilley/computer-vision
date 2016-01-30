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

// global state management
bool CudaGLviewer::s_globalStateInitialized = false;
std::mutex CudaGLviewer::s_GLlock; // gives 1 instance access at a time

// vertices for drawing image
GLfloat vertexArray[] = {
	// X Y U V
	-1.0f,  1.0f, 0.0f, 0.0f, // upper left
	 1.0f,  1.0f, 1.0f, 0.0f, // upper right
	-1.0f, -1.0f, 0.0f, 1.0f, // lower left

	-1.0f, -1.0f, 0.0f, 1.0f, // lower left
	 1.0f, -1.0f, 1.0f, 1.0f, // lower right
	 1.0f,  1.0f, 1.0f, 0.0f, // upper right
};

// call before instantiating this class
// returns 0 on success, 1 on failure
int CudaGLviewer::initGlobalState()
{
	// call only on first instance creation
	glfwSetErrorCallback(CudaGLviewer::cb_GLFWerror);

	// call only on first instance creation
	if(GL_FALSE == glfwInit())
	{
		// exit if we can't create any viewers properly
		std::cerr << "CudaGLviewer: error initializing GLFW" << std::endl;
		s_globalStateInitialized = false;
		return 1;
	}

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
GLuint CudaGLviewer::compileShaders(std::string vertexSourceFilename, std::string fragmentSourceFilename)
{
	// reference to shader to return
	GLuint shaderProgram = 0;

	// text inside each file, file handles
	std::string vertexSourceCode, fragmentSourceCode;
	std::ifstream vertexSourceFile(vertexSourceFilename);
	std::ifstream fragmentSourceFile(fragmentSourceFilename);

	// temp string for loading from file
	std::string tempStr;

	#ifdef DEBUG
		// compilation checking
		GLint compileSuccess = 0, validationSuccess = 0;
		char temp[256] = {0};
		GLint lenLinkInfoLog = 0;
		GLsizei charsInLog = 0;
	#endif

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
	m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
	m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	// bind their source code
	const GLchar* vtx = static_cast<const GLchar*>(vertexSourceCode.c_str());
	const GLchar* frag = static_cast<const GLchar*>(fragmentSourceCode.c_str());
	glShaderSource(m_vertexShader, 1, &vtx, NULL);
	glShaderSource(m_fragmentShader, 1, &frag, NULL);

	// compile shaders
	glCompileShader(m_vertexShader);
	glCompileShader(m_fragmentShader);

	#ifdef DEBUG
		// check compilation for error status

		// get status from OpenGL
		glGetShaderiv(m_vertexShader, GL_COMPILE_STATUS, &compileSuccess);

		if(0 == compileSuccess) // failed to compile
		{
			// print error message
			glGetShaderInfoLog(m_vertexShader, 256, NULL, temp);
			std::cerr << "vertex shader failed to compile" << std::endl;
			std::cerr << temp << std::endl;

			// destory allocated resources
			glDeleteShader(m_vertexShader);
			glDeleteShader(m_fragmentShader);
			glDeleteProgram(shaderProgram);

			// failure
			return 0;
		}
		else
		{
			// bind the shader to its program
			glAttachShader(shaderProgram, m_vertexShader);
		}

		// get status from OpenGL
		glGetShaderiv(m_fragmentShader, GL_COMPILE_STATUS, &compileSuccess);

		if(0 == compileSuccess) // failed to compile
		{
			// print error message
			glGetShaderInfoLog(m_fragmentShader, 256, NULL, temp);
			std::cerr << "fragment shader failed to compile" << std::endl;
			std::cerr << temp << std::endl;

			// destory allocated resources
			glDeleteShader(m_vertexShader);
			glDeleteShader(m_fragmentShader);
			glDeleteProgram(shaderProgram);

			// failure
			return 0;
		}
		else
		{
			// bind the shader to its program
			glAttachShader(shaderProgram, m_fragmentShader);
		}
	#else
		glAttachShader(shaderProgram, m_vertexShader);
		glAttachShader(shaderProgram, m_fragmentShader);
	#endif

	glBindFragDataLocation(shaderProgram, 0, "outColor");

	// link the shader program together
	glLinkProgram(shaderProgram);

	#ifdef DEBUG
		// let OpenGL validate shader program
		glValidateProgram(shaderProgram);

		// see if validation succeeded
		glGetProgramiv(shaderProgram, GL_VALIDATE_STATUS, &validationSuccess);

		std::cout << "shader validation " << ((0 != validationSuccess) ? "passed" : "failed") << std::endl;

		// check result of shader program compile
		glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &lenLinkInfoLog);

		// print the log if something went wrong
		if(0 < lenLinkInfoLog)
		{
			char* linkLog = static_cast<char*>(malloc(lenLinkInfoLog));
			glGetProgramInfoLog(shaderProgram, lenLinkInfoLog, &charsInLog, linkLog);
			std::cerr << "shader info log [" << lenLinkInfoLog << "]: " << linkLog << std::endl;
			free(linkLog);
		}
	#endif

	return shaderProgram;
}

// return 0 on success, 1 on failure
int CudaGLviewer::initGL()
{
	// disable deprecated functionality
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
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
	glErr();

	// NULL returned if context doesn't exist
	if(NULL == m_GLFWwindow)
	{
		m_isValid = false;
		return 1;
	}

	// set window close callback
	glfwSetWindowCloseCallback(m_GLFWwindow, cb_GLFWcloseWindow);

	// set key callback only if a queue was given
	if(m_pKeyEventQueue)
		glfwSetKeyCallback(m_GLFWwindow, cb_GLFWkeyEvent);

	// set framebuffer size callback
	glfwSetFramebufferSizeCallback(m_GLFWwindow, cb_GLFWframebufferSize);

	// set the current context for state-based OpenGL
	// lock must stay alive until end of calling scope
	GLcontextWrapper ctxWrapper;
	auto lock = captureGL(ctxWrapper);

	// initialize GLEW
	glewExperimental = GL_TRUE; // use current functionality
	glewInit();
	// expected to spit out "Unknown Error", so don't abort here
	glError(glGetError(), __FILE__, __LINE__, false);

	// attach this instance to the window to make life easier
	glfwSetWindowUserPointer(m_GLFWwindow, this);

	// set initial framebuffer size
	glViewport(0, 0, m_windowWidth, m_windowHeight);

	// V-SYNC enabled when not testing for performance
	#ifndef PERFORMANCE_TEST
		glfwSwapInterval(1);
	#else
		glfwSwapInterval(0);
	#endif

	// create the shader program
	m_shaderProgram = compileShaders(VERTEX_SHADER_FILENAME, FRAGMENT_SHADER_FILENAME);
	if(0 == m_shaderProgram)
	{
		std::cerr << "CudaGLviewer: error compiling shaders" << std::endl;
		return 1;
	}

	glErr();

	// success
	return 0;
}

// return 0 on success, 1 on failure
int CudaGLviewer::initCUDA()
{
	// main compute device will also be used for display
	cudaSetDevice(cudaPrimaryDevice);
	// cudaGLSetGLDevice(cudaPrimaryDevice);

	// success
	return 0;
}

// allocate the buffers needed for interop
// return 0 on success, 1 on failure
int CudaGLviewer::initBuffers()
{
	// get control of OpenGL
	GLcontextWrapper ctxWrapper;
	auto lock = captureGL(ctxWrapper);

	// OpenGL needs a vertex array
	glGenVertexArrays(1, &m_vertexArray);
	glBindVertexArray(m_vertexArray);

	// make the texure to copy CUDA output frames into
	glGenTextures(1, &m_cudaDestTexture);
	glBindTexture(GL_TEXTURE_2D, m_cudaDestTexture);

	// set its parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// memory allocation
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA/*8UI_EXT*/, m_imageWidth, m_imageHeight, 0, GL_RGBA/*_INTEGER_EXT*/, GL_UNSIGNED_BYTE, NULL);
	glErr();

	// register the texture with CUDA
	cudaErr(cudaGraphicsGLRegisterImage(&m_cudaDestResource, m_cudaDestTexture, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard));

	// map the resource (texure)
	cudaErr(cudaGraphicsMapResources(1, &m_cudaDestResource, 0));

	// bind the texture to a useable cudaArray
	cudaErr(cudaGraphicsSubResourceGetMappedArray(&m_cudaDestArray, m_cudaDestResource, 0, 0));

	// unmap texture (CUDA now has its own handle to the texture)
	cudaErr(cudaGraphicsUnmapResources(1, &m_cudaDestResource, 0));

	// generate a vertex buffer for drawing
	glGenBuffers(1, &m_vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexArray), vertexArray, GL_STATIC_DRAW);
	glErr();

	// bind vertices to shaders
	GLuint positionAttribute = glGetAttribLocation(m_shaderProgram, "position");
	glEnableVertexAttribArray(positionAttribute);
	// tells OpenGL where to find exact data elements
	glVertexAttribPointer(positionAttribute, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), 0);

	GLuint textureLocationAttribute = glGetAttribLocation(m_shaderProgram, "texcoord");
	glEnableVertexAttribArray(textureLocationAttribute);
	// tells OpenGL where to find exact data elements
	glVertexAttribPointer(textureLocationAttribute, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), reinterpret_cast<const GLvoid*>(2 * sizeof(GLfloat)));
	
	glErr();

	// set the color of a blank screen
	glClearColor(0.f, 0.f, 0.f, 1.f);

	// success
	return 0;
}

// return 0 on success, 1 on failure
int CudaGLviewer::freeResources()
{
	// destroy vertex data
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteVertexArrays(1, &m_vertexArray);
	glDeleteBuffers(1, &m_vertexBuffer);

	// destroy shaders
	glDeleteProgram(m_shaderProgram);
	glDeleteShader(m_vertexShader);
	glDeleteShader(m_fragmentShader);

	// destroy the texture used to display
	glDeleteTextures(1, &m_cudaDestTexture);

	glErr();
	m_cudaDestTexture = 0;

	// destroy the window attached to this instance
	glfwDestroyWindow(m_GLFWwindow);

	return 0;
}

CudaGLviewer::CudaGLviewer(unsigned imageWidth, unsigned imageHeight, std::string title, ConcurrentQueue<KeyEvent>* pKeyEventQueue): m_windowWidth(DEFAULT_WINDOW_WIDTH), m_windowHeight(DEFAULT_WINDOW_HEIGHT), m_imageWidth(imageWidth), m_imageHeight(imageHeight), m_windowTitle(title), m_pKeyEventQueue(pKeyEventQueue)
{
	// make sure global state is initialized
	if(!s_globalStateInitialized)
	{
		std::cerr << "CudaGLviewer: global state not initialized" << std::endl;
		std::cerr << "              please first call CudaGLviewer::initializeGlobalState()" << std::endl;
		m_isValid = false;
		return;
	}
	
	// initialize OpenGL state
	if(0 != initGL())
	{
		m_isValid = false;
		std::cerr << "initGL failed" << std::endl;
		return;
	}

	// initialize CUDA state
	if(0 != initCUDA())
	{
		m_isValid = false;
		std::cerr << "initCUDA failed" << std::endl;
		return;
	}

	// shared texture(s)
	if(0 != initBuffers())
	{
		m_isValid = false;
		std::cerr << "initBuffers failed" << std::endl;
		return;
	}

	// set texture coordinates without a size update
	cb_GLFWframebufferSize(m_GLFWwindow, m_windowWidth, m_windowHeight);

	m_isValid = true;
}

CudaGLviewer::~CudaGLviewer()
{
	// lock must stay alive until end of calling scope
	GLcontextWrapper ctxWrapper;
	auto lock = captureGL(ctxWrapper);

	// free all instance-attached resources
	freeResources();

	// not really necessary
	m_isValid = false;
}

// copy data to output texture
int CudaGLviewer::drawFrame(GPUFrame& inputCudaFrame)
{
	// don't let an invalid instance draw
	if(!good())
	{
		std::cerr << "CudaGLviewer::drawFrame called on bad instance" << std::endl;
		return 1; // failure
	}

	// window may need resizing
	if(m_windowResized)
	{
		glViewport(0, 0, m_windowWidth, m_windowHeight);
		m_windowResized = false; // reset the flag
	}

	// get complete ownership of GL to limit context switching
	// lock must stay alive until end of calling scope
	GLcontextWrapper ctxWrapper;
	auto lock = captureGL(ctxWrapper);

	// calculate the size of the display texture
	unsigned numTexels = m_imageWidth * m_imageHeight;
	unsigned numColorValues = 4 * numTexels;
	unsigned sizeTexture = sizeof(GLubyte) * numColorValues;

	// copy from output frame to array
	cudaErr(cudaMemcpy2DToArray(m_cudaDestArray, 0, 0, inputCudaFrame.data(), inputCudaFrame.pitch(), inputCudaFrame.width() * 4, inputCudaFrame.height(), cudaMemcpyDeviceToDevice));

	// bind the vertex array
	glBindVertexArray(m_vertexArray);

	// clear previous image
	glClear(GL_COLOR_BUFFER_BIT);

	// shader program to draw texture
	glUseProgram(m_shaderProgram);

	// draw new image
	glDrawArrays(GL_TRIANGLES, 0, 6);

	// swap front/back buffers
	glfwSwapBuffers(m_GLFWwindow);

	// unbind the vertex array to avoid interference
	glBindVertexArray(0);

	// check errors
	glErr();

	// success
	return 0;
}