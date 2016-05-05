#include "headers/DecoderCPU.h"
#include <iostream>

using namespace std;

AVCodecID DecoderCPU::getAVcodec(enum encoding e)
{
	switch(e)
	{
		case encoding_H264:
		return AV_CODEC_ID_H264;

		case encoding_MJPG:
		return AV_CODEC_ID_MJPEG;

		default:
		return AV_CODEC_ID_NONE;
	}
}

// specify format and output queue
DecoderCPU::DecoderCPU(enum encoding inputFormat, ConcurrentQueue<RawFrame>& q): Decoder(q, false)
{
	// init avcodec and member data
	avcodec_register_all();

	av_init_packet(&m_avpkt);

	m_codec = avcodec_find_decoder(getAVcodec(inputFormat));
	if(!m_codec)
	{
		cerr << "[" << __FILE__ << ":" << __LINE__ << "]" << " decoder not found" << endl;
		return;
	}

	m_codecContext = avcodec_alloc_context3(m_codec);
	if(!m_codecContext)
	{
		cerr << "[" << __FILE__ << ":" << __LINE__ << "]" << " could not allocate context" << endl;
		return;
	}

	if(m_codec->capabilities & AV_CODEC_CAP_TRUNCATED)
        m_codecContext->flags |= AV_CODEC_FLAG_TRUNCATED; // we do not send complete frames

	if (avcodec_open2(m_codecContext, m_codec, nullptr) < 0)
	{
    	cerr << "[" << __FILE__ << ":" << __LINE__ << "]" << " could not open codec" << endl;
	    return;
    }

    m_frame = av_frame_alloc();
    if(!m_frame)
    {
    	cerr << "[" << __FILE__ << ":" << __LINE__ << "]" << " could not allocate frame" << endl;
    	return;
    }
}

// pass encoded data to be decoded
int DecoderCPU::decodeFrame(const CodedFrame& frame)
{
	// fill packet struct
	m_avpkt.size = frame.size();
	m_avpkt.data = frame.data();

	// avcodec already returns desired error code
	int errCode = avcodec_send_packet(m_codecContext, &m_avpkt);

	// attempt to pull a frame out of the decoder
	if(!avcodec_receive_frame(m_codecContext, m_frame))
	{
		m_destQueue.push(RawFrameCPU(m_frame->data[0], m_frame->linesize[0],
			m_frame->width, m_frame->height, m_frame->linesize[0], m_frame->height, 0));
	}

	return errCode;
}

// signal to decoder stream is done
void DecoderCPU::endStream(void)
{
	// send empty packet
	m_avpkt.size = 0;
	m_avpkt.data = nullptr;
	avcodec_send_packet(m_codecContext, &m_avpkt);

	// flush entire internal buffer
	while(!avcodec_receive_frame(m_codecContext, m_frame))
	{
		m_destQueue.push(RawFrameCPU(m_frame->data[0], m_frame->linesize[0],
			m_frame->width, m_frame->height, m_frame->linesize[0], m_frame->height, 0));
	}

	// push eos frame to output queue
	m_destQueue.push(RawFrameCPU(true));
}