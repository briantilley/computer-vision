#include <iostream> // development

#include "headers/V4L2cam.h"

// specifying number of buffers is optional
V4L2cam::V4L2cam(std::string device, dataFormat format, unsigned& width, unsigned& height, unsigned _bufferCount)
{
	// keep the instance locked until creation finished
	std::unique_lock<std::mutex> mlock(m_cameraMutex);

	// member variables
	bufferCount = (0 != _bufferCount) ? _bufferCount : DEFAULT_BUFFER_COUNT;

	// local variables
	struct v4l2_capability device_caps;
	struct v4l2_format formatStruct;
	struct v4l2_ext_controls ext_ctrls;
	struct v4l2_requestbuffers request_bufs;

	// open device file (V4L2 is a C library)
	fileDescriptor = open(device.c_str(), O_RDWR);
	if(-1 == fileDescriptor)
	{
		std::cerr << "failed to open " << device << std::endl;
		m_isValid = false;
		return;
	}

	// query capabilites (suggested by V4L2)
	if(-1 == xioctl(fileDescriptor, VIDIOC_QUERYCAP, &device_caps))
	{
		std::cerr << "error while querying caps" << std::endl;
		m_isValid = false;
		return;
	}

	formatStruct.type                  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	formatStruct.fmt.pix.width         = width;
	formatStruct.fmt.pix.height        = height;
	formatStruct.fmt.pix.pixelformat   = format;
	formatStruct.fmt.pix.field         = V4L2_FIELD_NONE;

	// set video format
	if(-1 == xioctl(fileDescriptor, VIDIOC_S_FMT, &formatStruct))
	{
		std::cerr << "error while setting format" << std::endl;
		m_isValid = false;
		return;
	}

	// get and return the actual width and height
	videoWidth = width = formatStruct.fmt.pix.width;
	videoHeight = height = formatStruct.fmt.pix.height;

	ext_ctrls.count = 2;
	ext_ctrls.ctrl_class = V4L2_CTRL_CLASS_CAMERA;
	ext_ctrls.controls = new v4l2_ext_control[2];

	ext_ctrls.controls[0].id     = V4L2_CID_EXPOSURE_AUTO;
	ext_ctrls.controls[0].value  = V4L2_EXPOSURE_MANUAL;
	ext_ctrls.controls[1].id     = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
	ext_ctrls.controls[1].value  = 0;

	// disable auto exposure (limits framerate)
	if(-1 == xioctl(fileDescriptor, VIDIOC_S_EXT_CTRLS, &ext_ctrls))
	{
		std::cerr << "error while setting controls" << std::endl;
		m_isValid = false;
		return;
	}

	delete [] ext_ctrls.controls;

	request_bufs.count                 = bufferCount;
	request_bufs.type                  = formatStruct.type;
	request_bufs.memory                = V4L2_MEMORY_MMAP;

	// request input buffers for webcam data
	if(-1 == xioctl(fileDescriptor, VIDIOC_REQBUFS, &request_bufs))
	{
		std::cerr << "error while requesting buffers" << std::endl;
		m_isValid = false;
		return;
	}

	// get the actual number of buffers
	bufferCount = request_bufs.count;

	workingBuffer.type = request_bufs.type;
	workingBuffer.memory = request_bufs.memory;

	// buffers = (frameBuffer*)malloc(sizeof(frameBuffer) * bufferCount);
	buffers = new frameBuffer[bufferCount];

	// make an array of buffers in V4L2 and enqueue them to prepare for stream on
	for(int i = 0; i < bufferCount; ++i)
	{
		workingBuffer.index = i;
		if(-1 == xioctl(fileDescriptor, VIDIOC_QUERYBUF, &workingBuffer))
		{
			std::cerr << "error while querying buffer" << std::endl;
			m_isValid = false;
			return;
		}

		buffers[i].start = (byte*)mmap(NULL, workingBuffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, workingBuffer.m.offset);
		buffers[i].size = workingBuffer.length;

		if(MAP_FAILED == buffers[i].start)
		{
			std::cerr << "error mapping buffers " << std::endl;
			m_isValid = false;
			return;
		}

		if(-1 == xioctl(fileDescriptor, VIDIOC_QBUF, &workingBuffer))
		{
			std::cerr << "error while initial enqueuing" << std::endl;
			m_isValid = false;
			return;
		}
	}

	// only now is the camera object valid
	m_isValid = true;
}

V4L2cam::~V4L2cam()
{
	// shut off and free memory
	streamOff();
	for(int i = 0; i < bufferCount; ++i)
		munmap(buffers[i].start, buffers[i].size); // allocated with mmap()
	delete [] buffers;
}

float V4L2cam::getExposure(void)
{
	// make other operations wait
	std::unique_lock<std::mutex> mlock(m_cameraMutex);

	return 0.f;
}

// webcam on
int V4L2cam::streamOn(void)
{
	// make other operations wait
	std::unique_lock<std::mutex> mlock(m_cameraMutex);

	if(-1 == xioctl(fileDescriptor, VIDIOC_STREAMON, &workingBuffer.type))
	{
		std::cerr << "error while turning stream on" << std::endl;
		return 1;
	}

	// camera is on
	m_isOn = true;

	workingBuffer.index = 0;
	return 0;
}

// webcam off
int V4L2cam::streamOff(void)
{
	// make other operations wait
	std::unique_lock<std::mutex> mlock(m_cameraMutex);

	if(-1 == xioctl(fileDescriptor, VIDIOC_STREAMOFF, &workingBuffer.type))
	{
		std::cerr << "error while turning stream off" << std::endl;
		return 1;
	}

	// camera is off
	m_isOn = false;

	return 0;
}

int V4L2cam::setExposure(float exposure)
{
	// make other operations wait
	std::unique_lock<std::mutex> mlock(m_cameraMutex);

	return 1;
}

int V4L2cam::changeExposure(bool increase, float deltaExposure)
{
	// make other operations wait
	std::unique_lock<std::mutex> mlock(m_cameraMutex);

	return 1;
}

// get an encoded frame of data
CodedFrame V4L2cam::retrieveCodedFrame(void)
{
	// returned by this function
	CodedFrame returnFrame;
	unsigned timestamp;

	// make other operations wait
	std::unique_lock<std::mutex> mlock(m_cameraMutex);

	// if stream is off
	if(!isOn())
	{
		// give back an empty frame
		return CodedFrame();
	}

	// pull frame buffer out of v4l2's queue
	if(-1 == xioctl(fileDescriptor, VIDIOC_DQBUF, &workingBuffer))
	{
		std::cerr << "error while retrieving frame" << std::endl;
		m_isValid = false;
		mlock.unlock();
		streamOff();
		return CodedFrame();
	}

	// let other work happen during copy insided 'CodedFrame' ctor
	mlock.unlock();

	// create a copy of the data to return
	timestamp = workingBuffer.timestamp.tv_sec * 1000000;
	timestamp += workingBuffer.timestamp.tv_usec;
	returnFrame = CodedFrame(buffers[workingBuffer.index].start, workingBuffer.bytesused, timestamp);

	// re-lock for V4L2 access (unlocks on return)
	mlock.lock();

	// re-queue the buffer for v4l2
	if(-1 == xioctl(fileDescriptor, VIDIOC_QBUF, &workingBuffer))
	{
		std::cerr << "error while releasing buffer" << std::endl;
		m_isValid = false;
		mlock.unlock();
		streamOff();
		return CodedFrame();
	}

	// move forward in the queue
	++workingBuffer.index; workingBuffer.index %= bufferCount;

	// You left this line out. Don't forget again.
	return returnFrame;
}
