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

// development
#include <iostream>

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
	// creating multiple instances from the same pointer = undefined behavior
	CodedFrame(const byte* _data, unsigned _length, unsigned _timestamp)
	{
		// make a copy
		byte* newData = new byte[_length];
		memcpy(static_cast<void*>(newData), static_cast<const void*>(_data), _length);

		// lambda ensures allocation is fully freed
		m_data = std::shared_ptr<byte>(newData, [=](byte* p){ delete [] p; });
		m_length = _length;
		m_timestamp = _timestamp;
	}
	
	// copy constructor/assignment just duplicate state
	CodedFrame(const CodedFrame& toCopy) = default;
	CodedFrame& operator=(const CodedFrame& right) = default;
	
	// free copy of data
	~CodedFrame() = default;

	// check if frame is empty (possible use: error signaling)
	bool empty(void) const { return !static_cast<bool>(m_length); }

	unsigned size(void) const { return m_length; } // size of allocation
	const byte* raw_data(void) const { return m_data.get(); } // immutable information contained
	unsigned timestamp(void) const { return m_timestamp; } // capture time of frame

	// time between frames
	int operator-(const CodedFrame& right) const { return m_timestamp - right.m_timestamp; }
};

#endif
