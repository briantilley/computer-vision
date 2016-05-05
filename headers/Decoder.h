#ifndef DECODER_H
#define DECODER_H

#include "CodedFrame.h"
#include "RawFrame.h"
#include "ConcurrentQueue.h"

// user must specify format in constructor
enum encoding
{
	encoding_H264,
	encoding_MJPG,
	encoding_NONE
};

class Decoder
{
private:
	const bool m_isGPUdecoder;

protected:
	ConcurrentQueue<RawFrame>& m_destQueue;

	// dimensions of the video input
	size_t m_width, m_height;

public:
	Decoder(ConcurrentQueue<RawFrame>& destQueue, bool isGPUdecoder): m_destQueue(destQueue), m_isGPUdecoder(isGPUdecoder) { }

	// no copying allowed
	Decoder(Decoder&) = delete;

	// access data
	size_t width(void) const { return m_width; }
	size_t height(void) const { return m_height; }

	virtual int decodeFrame(const CodedFrame& frame) = 0; // decoded frame will be put in destQueue
	virtual void endStream(void) = 0; // signal end of stream to the decoder

	bool isGPUdecoder(void) const { return m_isGPUdecoder; }
};

#endif