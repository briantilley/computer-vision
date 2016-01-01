#include <iostream> // development

#include "headers/V4L2cam.h"

// specifying number of buffers is optional
V4L2cam::V4L2cam(std::string device, dataFormat format, unsigned& width, unsigned& height, unsigned _bufferCount)
{
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
		exit(1);
	}

	// query capabilites (suggested by V4L2)
	if(-1 == xioctl(fileDescriptor, VIDIOC_QUERYCAP, &device_caps))
	{
		std::cerr << "error while querying caps" << std::endl;
		exit(1);
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
		exit(1);
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
		exit(1);
	}

	delete [] ext_ctrls.controls;

	request_bufs.count                 = bufferCount;
	request_bufs.type                  = formatStruct.type;
	request_bufs.memory                = V4L2_MEMORY_MMAP;

	// request input buffers for webcam data
	if(-1 == xioctl(fileDescriptor, VIDIOC_REQBUFS, &request_bufs))
	{
		std::cerr << "error while requesting buffers" << std::endl;
		exit(1);
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
			exit(1);
		}

		buffers[i].start = (byte*)mmap(NULL, workingBuffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, workingBuffer.m.offset);
		buffers[i].size = workingBuffer.length;

		if(MAP_FAILED == buffers[i].start)
		{
			std::cerr << "error mapping buffers " << std::endl;
			exit(1);
		}

		if(-1 == xioctl(fileDescriptor, VIDIOC_QBUF, &workingBuffer))
		{
			std::cerr << "error while initial enqueuing" << std::endl;
			exit(1);
		}
	}
}

// V4L2cam::V4L2cam(const V4L2cam&)
// {

// }

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
	return 0.f;
}

// webcam on
int V4L2cam::streamOn(void)
{
	if(-1 == xioctl(fileDescriptor, VIDIOC_STREAMON, &workingBuffer.type))
	{
		std::cerr << "error while turning stream on" << std::endl;
		return 1;
	}

	workingBuffer.index = 0;
	return 0;
}

// webcam off
int V4L2cam::streamOff(void)
{
	if(-1 == xioctl(fileDescriptor, VIDIOC_STREAMOFF, &workingBuffer.type))
	{
		std::cerr << "error while turning stream off" << std::endl;
		return 1;
	}

	return 0;
}

int V4L2cam::setExposure(float exposure)
{
	return 1;
}

int V4L2cam::changeExposure(bool increase, float deltaExposure)
{
	return 1;
}

// get an encoded frame of data
CodedFrame V4L2cam::retrieveCodedFrame(void)
{
	// returned by this function
	CodedFrame returnFrame;
	float timestamp;

	// pull frame buffer out of v4l2's queue
	if(-1 == xioctl(fileDescriptor, VIDIOC_DQBUF, &workingBuffer))
	{
		std::cerr << "error while retrieving frame" << std::endl;
		exit(1);
	}

	// create a copy of the data to return
	timestamp = workingBuffer.timestamp.tv_sec * 1000000;
	timestamp += workingBuffer.timestamp.tv_usec;
	returnFrame = CodedFrame(buffers[workingBuffer.index].start, workingBuffer.bytesused, timestamp);

	// re-queue the buffer for v4l2
	if(-1 == xioctl(fileDescriptor, VIDIOC_QBUF, &workingBuffer))
	{
		std::cerr << "error while releasing buffer" << std::endl;
		exit(1);
	}

	// move forward in the queue
	++workingBuffer.index; workingBuffer.index %= bufferCount;

	// You left this line out. Don't forget again.
	return returnFrame;
}