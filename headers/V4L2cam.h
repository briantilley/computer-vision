#ifndef V4L2_CAM_H
#define V4L2_CAM_H

// thread safety
#include <thread>
#include <mutex>

// V4L2 includes
#include <linux/videodev2.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/mman.h>

// helper class
#include "CodedFrame.h"

// more explicit
typedef uint8_t byte;

// user must specify format in constructor
enum dataFormat
{
	h264 = V4L2_PIX_FMT_H264
};

#define DEFAULT_BUFFER_COUNT 8

class V4L2cam
{
private:

	// signal valid input object w/ implicit conversion
	bool m_isValid = false;
	std::mutex m_cameraMutex;

	// only V4L2 needs this type
	typedef struct _frameBuffer
	{
		byte* start;
		unsigned size; // stored for call to munmap() on deletion
	} frameBuffer;

	int fileDescriptor;
	unsigned videoWidth, videoHeight; // dimensions
	struct v4l2_buffer workingBuffer; // let this track index in v4l2's queue
	frameBuffer* buffers;
	unsigned bufferCount;

	// signal to any threads that the stream is off
	bool m_isOn;

	// wrapper for ioctl function
	// only this class needs access
	inline int xioctl(int fd, int req, void* pArgs)
	{
		// ioctl returns error codes
		int errCode = 0;

		// keep calling ioctl until result is obtained
		do errCode = ioctl(fd, req, pArgs);
		while(-1 == errCode && EINTR == errno);

		return errCode;
	}

public:

	// constructors
	V4L2cam(std::string device, dataFormat format, unsigned& width, unsigned& height, unsigned numBuffers=DEFAULT_BUFFER_COUNT);

	~V4L2cam();

	// accessors
	int getWidth(void) const { return videoWidth; }
	int getHeight(void) const { return videoHeight; }
	float getExposure(void); // return exposure value in seconds
	bool isOn(void) const { return m_isOn; }
	
	// indicate good stream
	operator bool() const { return m_isValid; } // implicit conversion
	bool good() const { return m_isValid; }

	// mutators
	int streamOn(void); // start v4l2 stream
	int streamOff(void); // stop v4l2 stream
	int setExposure(float exposure); // set to exposure in seconds
	int changeExposure(bool increase, float deltaExposure); // lengthen/shorten by deltaExposure

	// utilities
	CodedFrame retrieveCodedFrame(void);
};

#endif
