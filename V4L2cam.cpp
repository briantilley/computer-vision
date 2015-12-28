#include <iostream> // development

// V4L2 includes
#include <linux/videodev2.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/mman.h>

// explicit datatype names (int32_t, uint8_t, etc)
#include <unistd.h>

#include "headers/V4L2cam.h"

V4L2cam::V4L2cam(std::string device, dataFormat format, int& width, int& height, int numBuffers)
{
	// local variables
	struct v4l2_capability device_caps;
	struct v4l2_format formatStruct;
	struct v4l2_ext_controls ext_ctrls;
	struct v4l2_requestbuffers request_bufs;
	struct v4l2_buffer buffer;

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
	ext_ctrls.controls = (v4l2_ext_control*)malloc(2 * sizeof(v4l2_ext_control));

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

	request_bufs.count                 = numBuffers;
	request_bufs.type                  = formatStruct.type;
	request_bufs.memory                = V4L2_MEMORY_MMAP;

	// request input buffers for webcam data
	if(-1 == xioctl(fileDescriptor, VIDIOC_REQBUFS, &request_bufs))
	{
		std::cerr << "error while requesting buffers" << std::endl;
		exit(1);
	}

	// get the actual number of buffers
	numBuffers = request_bufs.count;

	buffer.type = request_bufs.type;
	buffer.memory = request_bufs.memory;

	buffers = (frameBuffer*)malloc(sizeof(frameBuffer) * numBuffers);

	// make an array of buffers in V4L2 and enqueue them to prepare for stream on
	for(int i = 0; i < numBuffers; ++i)
	{
		buffer.index = i;
		if(-1 == xioctl(fileDescriptor, VIDIOC_QUERYBUF, &buffer))
		{
			std::cerr << "error while querying buffer" << std::endl;
			exit(1);
		}

		buffers[i].start = (byte*)mmap(NULL, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, buffer.m.offset);

		if(MAP_FAILED == buffers[i].start)
		{
			std::cerr << "error mapping buffers " << std::endl;
			exit(1);
		}

		if(-1 == xioctl(fileDescriptor, VIDIOC_QBUF, &buffer))
		{
			std::cerr << "error while initial enqueuing" << std::endl;
			exit(1);
		}
	}
}

V4L2cam::V4L2cam(const V4L2cam&)
{

}

V4L2cam::~V4L2cam()
{

}

float V4L2cam::getExposure(void)
{

}

int V4L2cam::setExposure(float exposure)
{

}

int V4L2cam::changeExposure(bool increase, float deltaExposure)
{

}

int V4L2cam::retrieveCodedFrame(/*callback fxn?*/)
{

}