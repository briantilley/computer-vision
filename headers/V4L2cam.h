#ifndef V4L2_CAM_H
#define V4L2_CAM_H

// more explicit
typedef uint8_t byte;

// user must specify format in constructor
typedef enum _dataFormat
{
	h264 = V4L2_PIX_FMT_H264
} dataFormat;

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
	int videoWidth;
	int videoHeight;
	frameBuffer* buffers;

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
	V4L2cam(std::string device, dataFormat format, int& width, int& height, int numBuffers);
	V4L2cam(const V4L2cam&); // copy

	~V4L2cam();

	// accessors
	int getWidth(void) { return videoWidth; }
	int getHeight(void) { return videoHeight; }
	float getExposure(void); // return exposure value in seconds

	// mutators
	int setExposure(float exposure); // set to exposure in seconds
	int changeExposure(bool increase, float deltaExposure); // lengthen/shorten by deltaExposure

	// utilities
	int retrieveCodedFrame(/*callback fxn?*/);
};

#endif