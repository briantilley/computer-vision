#ifndef DECODER_GPU_H
#define DECODER_GPU_H

#include "Decoder.h"
#include "RawFrameGPU.h"

class DecoderGPU: public Decoder
{
private:

	// callback functions will have to use pUserData to get back to these
	CUvideodecoder m_decoderHandle = 0;
	CUvideoparser m_parserHandle = 0;
	unsigned m_width = 0, m_height = 0; // dimensions of the video stream

	// cuda driver is asynchronous with respect to decoding
	// this handles all issues with threads and concurrency
	ConcurrentQueue<GPUFrame>& m_outputQueue;

	// keep track of frames left in NVIDIA-managed queue
	int m_currentDecodeGap = 0;

	// hold onto a signaled end of stream to follow
	// the last frame with an empty eos one
	bool m_eosFlag = false;

	// CUDA driver is littered with global state (ew)
	// this allows NVdecoder instances across multiple threads
	static CUvideoctxlock s_lock; // pass this to decoder objects
	static bool s_lockInitialized;

	// need to keep track for destruction of context lock
	static int s_instanceCount;
	static bool s_globalStateInitialized;

	// callbacks for decoder
	static int sequence_callback(void*, CUVIDEOFORMAT*);
	static int decode_callback(void*, CUVIDPICPARAMS*);
	static int output_callback(void*, CUVIDPARSERDISPINFO*);

public:

	// specify that this is a GPU-implemented decoder
	DecoderGPU(dataFormat inputFormat, ConcurrentQueue<RawFrameGPU>& q);

	// disable copying
	DecoderGPU(const DecoderGPU&) = delete;
	DecoderGPU& operator=(const DecoderGPU&) = delete;
};

#endif