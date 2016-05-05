#ifndef RAW_FRAME_H
#define RAW_FRAME_H

#include <memory>

// base class for CPU and GPU implementations
class RawFrame
{
private:
	bool m_residesInGPUmemory; // indicates where data is located

protected:
	std::shared_ptr<void> m_data;
	size_t m_pitch;
	size_t m_width;
	size_t m_height;
	unsigned m_timestamp;
	bool m_endOfStream; // signals end of video stream

public:
	// let all members remain in the empty state
	RawFrame(): m_pitch(0), m_width(0), m_height(0), m_timestamp(0), m_endOfStream(false) { }

	// allow for end of stream object
	RawFrame(bool eos): m_pitch(0), m_width(0), m_height(0), m_timestamp(0), m_endOfStream(eos) { }

	// made for getting new allocations
	// GPUorCPU: false=CPU, true=GPU
	RawFrame(unsigned imageWidth, unsigned imageHeight, unsigned allocationCols, unsigned allocationRows, unsigned timestamp, bool GPUorCPU,
		bool eos=false):m_pitch(0), m_width(imageWidth), m_height(imageHeight), m_timestamp(timestamp), m_residesInGPUmemory(GPUorCPU), m_endOfStream(eos) { }

	// status of frame
	bool empty(void) const { return m_width && m_height; }

	// expose member data
	void* data(void) const { return m_data.get(); }
	size_t pitch(void) const {return m_pitch; }
	size_t width(void) const { return m_width; }
	size_t height(void) const { return m_height; }
	unsigned timestamp(void) const { return m_timestamp; }
	bool endOfStream(void) const { return m_endOfStream; }

	bool isGPUframe(void) const { return m_residesInGPUmemory; }
};

#endif