#ifndef V4L2_CAM_H
#define V4L2_CAM_H

// V4L2 includes
#include <linux/videodev2.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <sys/mman.h>

// more explicit
typedef uint8_t byte;

// user must specify format in constructor
enum dataFormat
{
	h264 = V4L2_PIX_FMT_H264
};

#define DEFAULT_BUFFER_COUNT 8

#include <string.h>

// helper class (allows seamless temporary dynamic memory allocation)
class CodedFrame
{
private:

	byte* data = nullptr;
	unsigned length = 0;

public:

	CodedFrame(void) { } // leave initialized as empty frame

	// copy raw data and wrap in object
	CodedFrame(const byte* _data, unsigned _length)
	{
		// store size and allocate space
		length = _length;
		data = new byte[length];

		// C-style copy is OK because data is only raw bytes
		memcpy(reinterpret_cast<void*>(data), reinterpret_cast<const void*>(_data), length);
	}
	
	// copy constructor (same behavior as above)
	CodedFrame(const CodedFrame& toCopy)
	{
		// store size and allocate space
		length = toCopy.size();
		data = new byte[length];

		// C-style copy is OK because data is only raw bytes
		memcpy(reinterpret_cast<void*>(data), reinterpret_cast<const void*>(toCopy.raw_data()), length);
	}

	// copy assignment
	void operator=(const CodedFrame& right)
	{
		// clear existing data
		delete [] data;

		// store size and allocate space
		length = right.size();
		data = new byte[length];

		// C-style copy is OK because data is only raw bytes
		memcpy(reinterpret_cast<void*>(data), reinterpret_cast<const void*>(right.raw_data()), length);
	}
	
	// free copy of data
	~CodedFrame()
	{ delete [] data; }

	// check if frame is empty (possible use: error signaling)
	bool empty(void) const { return static_cast<bool>(length); }

	unsigned size(void) const { return length; }
	const byte* raw_data(void) const { return data; }
};

class V4L2cam
{
private:

	// only V4L2 needs this type
	typedef struct _frameBuffer
	{
		byte* start;
		unsigned length;
	} frameBuffer;

	int fileDescriptor;
	unsigned videoWidth, videoHeight; // dimensions
	struct v4l2_buffer workingBuffer; // let this track index in v4l2's queue
	frameBuffer* buffers;
	unsigned bufferCount;

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
	// V4L2cam(const V4L2cam&); // copy

	~V4L2cam();

	// accessors
	int getWidth(void) { return videoWidth; }
	int getHeight(void) { return videoHeight; }
	float getExposure(void); // return exposure value in seconds

	// mutators
	int streamOn(void); // start v4l2 stream
	int streamOff(void); // stop v4l2 stream
	int setExposure(float exposure); // set to exposure in seconds
	int changeExposure(bool increase, float deltaExposure); // lengthen/shorten by deltaExposure

	// utilities
	CodedFrame retrieveCodedFrame(void);
};

#endif