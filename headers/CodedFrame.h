#ifndef CODED_FRAME_H
#define CODED_FRAME_H

/*
 * Helper class for V4L2cam
 * allows for easier sharing of frame buffers
 *
 * contained data cannot be guaranteed to
 * persist after associated object is destroyed
 */

#include <memory>
#include <cstring>

// more explicit
typedef uint8_t byte;

// helper class (allows seamless temporary dynamic memory allocation)
class CodedFrame
{
private:

	std::shared_ptr<byte> m_data;
	unsigned m_length = 0;
	unsigned m_timestamp = 0; // time value in microseconds (absolute value is arbitrary)

public:

	CodedFrame(void) { } // leave initialized as empty frame

	// copy raw data and wrap in object
	CodedFrame(const byte* _data, unsigned _length, unsigned _timestamp)
	{
		// make a copy
		byte* newData = new byte[_length];
		memcpy(static_cast<void*>(newData), static_cast<const void*>(_data), _length);

		// 2nd arg gives proper way to delete arrays when data is gone for good
		m_data = std::shared_ptr<byte>(newData, std::default_delete<byte[]>());
		m_length = _length;
		m_timestamp = _timestamp;
	}
	
	// copy constructor (same behavior as above)
	CodedFrame(const CodedFrame& toCopy)
	{
		m_data = toCopy.m_data;
		m_length = toCopy.m_length;
		m_timestamp = toCopy.m_timestamp;
	}

	// copy assignment
	void operator=(const CodedFrame& right)
	{
		m_data = right.m_data;
		m_length = right.m_length;
		m_timestamp = right.m_timestamp;
	}
	
	// free copy of data
	~CodedFrame() = default;

	// check if frame is empty (possible use: error signaling)
	bool empty(void) const { return static_cast<bool>(m_length); }

	unsigned size(void) const { return m_length; } // size of allocation
	const byte* raw_data(void) const { return m_data.get(); } // immutable information contained
	unsigned timestamp(void) const { return m_timestamp; } // capture time of frame
};

#endif
