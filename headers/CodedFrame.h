#ifndef CODED_FRAME_H
#define CODED_FRAME_H

/*
 * Helper class for V4L2cam
 * allows for easier sharing of frame buffers
 *
 * contained data cannot be guaranteed to
 * persist after associated object is destroyed
 */

#include <map>

// more explicit
typedef uint8_t byte;

// helper class (allows seamless temporary dynamic memory allocation)
class CodedFrame
{
private:

	// byte* data = nullptr;
	std::map<const byte*, int>::iterator allocationIterator;
	unsigned length = 0;
	unsigned m_timestamp = 0; // time value in microseconds (absolute value is arbitrary)

	// track the number of instances associated with each byte array
	static std::map<const byte*, int> allocations;

public:

	CodedFrame(void) { } // leave initialized as empty frame

	// copy raw data and wrap in object
	CodedFrame(const byte* _data, unsigned _length, float _timestamp);
	
	// copy constructor (same behavior as above)
	CodedFrame(const CodedFrame& toCopy);

	// copy assignment
	void operator=(const CodedFrame& right);
	
	// free copy of data
	~CodedFrame();

	// check if frame is empty (possible use: error signaling)
	bool empty(void) const { return static_cast<bool>(length); }

	unsigned size(void) const { return length; } // size of allocation
	const byte* raw_data(void) const { return allocationIterator->first; } // immutable information contained
	float timestamp(void) const { return m_timestamp; } // capture time of frame
};

#endif