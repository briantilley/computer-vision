#ifndef NV_DECODER_H
#define NV_DECODER_H

// wrapper for NVIDIA's C-style video decoder

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

	// cuda driver is asynchronous with respect to decoding
	// this handles all issues with threads and concurrency
	ConcurrentQueue<GPUFrame>& m_outputQueue;

	// CUDA driver is littered with global state (ew)
	// this allows NVdecoder instances across multiple threads
	static CUvideoctxlock s_lock; // pass this to decoder objects
	static bool s_lockInitialized;

	// need to keep track for destruction of context lock
	static int s_instanceCount;

public:

	NVdecoder(ConcurrentQueue<GPUFrame>& outputQueue);
	~NVdecoder();

	// access/mutation (technically breaks encapsulation, tsk tsk)
	CUvideodecoder& CUdecoder() { return m_decoderHandle; }

	// access (mutation happens inside class)
	CUvideoparser CUparser() const { return m_parserHandle; }
	CUvideoctxlock vidLock() const { return s_lock; }

	// utilities
	int decodeFrame(const CodedFrame& frame, CUvideopacketflags flags=static_cast<CUvideopacketflags>(0));
	void pushFrame(GPUFrame& toPush) { m_outputQueue.push(toPush); }
	GPUFrame popFrame(void);
};

#endif
