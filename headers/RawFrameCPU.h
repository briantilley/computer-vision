#include "RawFrame.h"
#include <cstring>

// client code is trusted not to modify the contents of the frame
class RawFrameCPU: public RawFrame
{
public:

	// let all members remain in the empty state
	RawFrameCPU(): RawFrame() { }

	// allow for end of stream object
	RawFrameCPU(bool eos): RawFrame(eos) { }

	// make an entirely new allocation
	template<typename T>
	RawFrameCPU(unsigned imageWidth, unsigned imageHeight, T* pitches, RawFrame::format format, unsigned timestamp,
		bool eos=false): RawFrame(imageWidth, imageHeight, pitches, format, timestamp, false, eos)
	{
		// get space on the heap
		uint8_t* newAllocation = nullptr;
		size_t allocationSize = 0;
		switch(format)
		{
			case RAW_FRAME_FORMAT_YUVJ422P:
				allocationSize = (pitches[0] + pitches[1] + pitches[2]) * m_height;
			break;

			case RAW_FRAME_FORMAT_NV12:
				allocationSize = (pitches[0] * 3 * m_height) / 2;
			break;

			default:
			break;
		}

		newAllocation = new uint8_t[allocationSize];

		// // track allocation with the shared_ptr
		m_data = std::shared_ptr<uint8_t>(newAllocation, [=](uint8_t* p){ delete [] p; });
	}

	// copy from given location
	template<typename T>
	RawFrameCPU(uint8_t* data, unsigned imageWidth, unsigned imageHeight, T* pitches, RawFrame::format format, unsigned timestamp,
		bool eos=false): RawFrameCPU(imageWidth, imageHeight, pitches, format, timestamp, eos)
	{
		// copy into a more permanent chunk of memory allocated by above ctor
		size_t allocationSize = 0;
		switch(format)
		{
			case RAW_FRAME_FORMAT_YUVJ422P:
				allocationSize = (pitches[0] + pitches[1] + pitches[2]) * m_height;
			break;

			case RAW_FRAME_FORMAT_NV12:
				allocationSize = (pitches[0] * 3 * m_height) / 2;
			break;

			default:
			break;
		}

		memcpy(m_data.get(), data, allocationSize);
	}

	// let C++ copy all member data
	RawFrameCPU(const RawFrameCPU&) = default;
	RawFrameCPU& operator=(const RawFrameCPU&) = default;

	~RawFrameCPU() = default;

	// time between frames
	int operator-(const RawFrameCPU& right) const { return m_timestamp - right.m_timestamp; }
};

