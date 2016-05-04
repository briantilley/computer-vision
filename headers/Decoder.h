#ifndef DECODER_H
#define DECODER_H

#include "CodedFrame.h"
#include "RawFrame.h"
#include "ConcurrentQueue.h"

// user must specify format in constructor
enum dataFormat
{
	H264,
	MJPG
};

class Decoder
{
private:
	const bool m_isGPUdecoder;

protected:
	ConcurrentQueue<RawFrame>& m_destQueue;

public:
	Decoder(ConcurrentQueue<RawFrame>& destQueue, bool isGPUdecoder): m_destQueue(destQueue), m_isGPUdecoder(isGPUdecoder) { }

	// no copying allowed
	Decoder(Decoder&) = delete;

	// access data
	size_t width(void) const { return m_width; }
	size_t height(void) const { return m_height; }

	int decodeFrame(const CodedFrame& frame); // decoded frame will be put in destQueue
	void endStream(void); // signal end of stream to the decoder

	bool isGPUdecoder(void) const { return m_isGPUdecoder; }
};

#endif