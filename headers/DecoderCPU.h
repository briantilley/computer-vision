#ifndef DECODER_CPU_H
#define DECODER_CPU_H

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>
}

#include "Decoder.h"
#include "RawFrameCPU.h"

class DecoderCPU: public Decoder
{
private:

	// libavcodec data
	AVCodec* m_codec = nullptr;
	AVCodecContext* m_codecContext = nullptr;
	AVFrame* m_frame = nullptr;
	AVPacket m_avpkt;

	// helper functions
	static AVCodecID getAVcodec(enum encoding);

public:

	// specify in constructor that this is a GPU-implemented decoder
	DecoderCPU(enum encoding inputFormat, ConcurrentQueue<RawFrame>& q);

	// disable copying
	DecoderCPU(const DecoderCPU&) = delete;
	DecoderCPU& operator=(const DecoderCPU&) = delete;

	int decodeFrame(const CodedFrame& frame); // decoded frame will be put in destQueue
	void endStream(void); // signal end of stream to the decoder

};

#endif