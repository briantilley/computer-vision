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
	RawFrameCPU(unsigned imageWidth, unsigned imageHeight, unsigned allocationCols, unsigned allocationRows,
			 unsigned timestamp, bool eos=false): RawFrame(imageWidth, imageHeight, allocationCols, allocationRows,
			 timestamp, eos)
	{
		// get space on the heap
		uint8_t* newAllocation;
		newAllocation = new uint8_t[allocationRows * allocationCols];

		// // track allocation with the shared_ptr
		m_data = std::shared_ptr<uint8_t>(newAllocation, [=](uint8_t* p){ delete [] p; });
	}

	// copy from given location
	RawFrameCPU(uint8_t* data, unsigned pitch,
			 unsigned imageWidth, unsigned imageHeight, unsigned allocationCols, unsigned allocationRows,
			 unsigned timestamp, bool eos=false): RawFrameCPU(imageWidth, imageHeight, allocationCols, allocationRows, timestamp)
	{
		// copy into a more permanent chunk of memory allocated by above ctor
		memcpy(m_data.get(), data, allocationRows * allocationCols);
	}

	// let C++ copy all member data
	RawFrameCPU(const RawFrameCPU&) = default;
	RawFrameCPU& operator=(const RawFrameCPU&) = default;

	~RawFrameCPU() = default;

	// time between frames
	int operator-(const RawFrameCPU& right) const { return m_timestamp - right.m_timestamp; }
};