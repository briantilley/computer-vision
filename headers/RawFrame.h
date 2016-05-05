#ifndef RAW_FRAME_H
#define RAW_FRAME_H

#include <memory>
#include <cstring>

// base class for CPU and GPU implementations
class RawFrame
{
public:
	enum format
	{
		RAW_FRAME_FORMAT_NONE = 0,
		RAW_FRAME_FORMAT_YUVJ422P,
		RAW_FRAME_FORMAT_NV12
	};

private:
	bool m_residesInGPUmemory; // indicates where data is located

protected:
	std::shared_ptr<void> m_data;
	size_t m_pitches[8] = {0};
	size_t m_width;
	size_t m_height;
	format m_format;
	unsigned m_timestamp;
	bool m_endOfStream; // signals end of video stream

public:
	// let all members remain in the empty state
	RawFrame(): m_width(0), m_height(0), m_timestamp(0), m_endOfStream(false) { }

	// allow for end of stream object
	RawFrame(bool eos): m_width(0), m_height(0), m_timestamp(0), m_endOfStream(eos) { }

	// made for getting new allocations
	// GPUorCPU: false=CPU, true=GPU
	template<typename T>
	RawFrame(unsigned imageWidth, unsigned imageHeight, T* pitches, format fmt, unsigned timestamp, bool GPUorCPU,
		bool eos=false):m_width(imageWidth), m_height(imageHeight), m_timestamp(timestamp), m_residesInGPUmemory(GPUorCPU), m_endOfStream(eos)
	{
		switch(fmt)
		{
			case RAW_FRAME_FORMAT_YUVJ422P:
				memcpy(m_pitches, pitches, 3 * sizeof(size_t));
			break;

			case RAW_FRAME_FORMAT_NV12:
				m_pitches[0] = pitches[0];
			break;

			default:
			break;
		}
	}

	// status of frame
	bool empty(void) const { return m_width && m_height; }

	// expose member data
	void* data(void) const { return m_data.get(); }
	const size_t* pitches(void) const {return m_pitches; }
	size_t width(void) const { return m_width; }
	size_t height(void) const { return m_height; }
	unsigned timestamp(void) const { return m_timestamp; }
	bool endOfStream(void) const { return m_endOfStream; }

	bool isGPUframe(void) const { return m_residesInGPUmemory; }
};

#endif