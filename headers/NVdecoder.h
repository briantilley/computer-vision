#ifndef NV_DECODER_H
#define NV_DECODER_H

// wrapper for NVIDIA's C-style video decoder
// need a way to cleanly signal discontinuities

#include <nvcuvid.h>
#include "CodedFrame.h"
#include "GPUFrame.h"
#include "ConcurrentQueue.h"

// development
#include <iostream>

// cuda driver error checking
#define cuErr(err) cuError(err, __FILE__, __LINE__)
inline void cuError(CUresult err, const char file[], unsigned line, bool abort=true)
{
    if(CUDA_SUCCESS != err)
    {
    	const char* str;
    	cuGetErrorName(err, &str);

        std::cerr << "[" << file << ":" << line << "] ";
        std::cerr << str << std::endl;
        if(abort) exit(err);
    }
}

#define DEFAULT_DECODE_SURFACES 8 // number of buffers to use while decoding
#define DEFAULT_CLOCK_RATE 0 // not sure what effect this has
#define DEFAULT_ERROR_THRESHOLD 10 // % corruption allowed in output stream
#define DEFAULT_DECODE_GAP 1 // number of frames between decode and mapping for output
#define DEFAULT_OUTPUT_SURFACES 8 // number of output buffers

// more explicit
typedef uint8_t byte;

class NVdecoder
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

	NVdecoder(ConcurrentQueue<GPUFrame>& outputQueue);
	~NVdecoder();

	// disable copying
	NVdecoder(const NVdecoder&) = delete;
	NVdecoder& operator=(const NVdecoder&) = delete;

	// access (mutation happens inside class)
	unsigned videoWidth(void) const { return m_width; }
	unsigned videoHeight(void) const { return m_height; }
	bool empty() const { return m_currentDecodeGap <= 0; }
	int decodeGap() const { return m_currentDecodeGap; }

	// utilities
	// too many calls to decodeFrame without popping the queue overflows GPU memory
	int decodeFrame(const CodedFrame& frame);
	void signalEndOfStream(void); // use this after sending the last decoded frame for the stream
};

#endif
