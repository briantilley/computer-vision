#include "headers/NVdecoder.h"
#include "headers/constants.h"
#include <cstring>

extern unsigned cudaPrimaryDevice;
extern unsigned cudaSecondaryDevice;

// change callbacks to static methods and fix encapsulation issues

// parser works on basis of callback functions
int NVdecoder::sequence_callback(void *pUserData, CUVIDEOFORMAT* pVidFmt)
{
	// stitch C API and C++ together
	NVdecoder* pInstance = reinterpret_cast<NVdecoder*>(pUserData);

	CUVIDDECODECREATEINFO dci;

	// only create the decoder if it doesn't already exist
	if(0 == pInstance->m_decoderHandle)
	{
		// use pVidFmt to fill packaged info
		dci.ulWidth             = pVidFmt->coded_width;
		dci.ulHeight            = pVidFmt->coded_height;
		dci.ulNumDecodeSurfaces = DEFAULT_DECODE_SURFACES;
		dci.CodecType           = cudaVideoCodec_H264; // magic value
		dci.ChromaFormat        = cudaVideoChromaFormat_422; // magic value
		dci.ulCreationFlags     = cudaVideoCreate_Default; // magic value

		memset(dci.Reserved1, 0, sizeof(dci.Reserved1));

		dci.display_area.left   = pVidFmt->display_area.left;
		dci.display_area.top    = pVidFmt->display_area.top;
		dci.display_area.right  = pVidFmt->display_area.right;
		dci.display_area.bottom = pVidFmt->display_area.bottom;

		dci.OutputFormat        = cudaVideoSurfaceFormat_NV12; // only fmt supported
		dci.DeinterlaceMode     = cudaVideoDeinterlaceMode_Adaptive; // magic value
		dci.ulTargetWidth       = dci.display_area.right;
		dci.ulTargetHeight      = dci.display_area.bottom;
		dci.ulNumOutputSurfaces = DEFAULT_OUTPUT_SURFACES;

		dci.vidLock             = pInstance->s_lock; // come back to this for multithreading
		dci.target_rect.left    = 0;
		dci.target_rect.top     = 0;
		dci.target_rect.right   = dci.ulTargetWidth;
		dci.target_rect.bottom  = dci.ulTargetHeight;

		memset(dci.Reserved2, 0, sizeof(dci.Reserved2));

		// create the decoder
		cuErr(cuvidCreateDecoder(&pInstance->m_decoderHandle, &dci));
	}

	// fill width and height of decoder
	pInstance->m_width = pVidFmt->display_area.right;
	pInstance->m_height = pVidFmt->display_area.bottom;

	// return value of 1 indicates success
	return 1;
}

int NVdecoder::decode_callback(void *pUserData, CUVIDPICPARAMS* pPicParams)
{
	// stitch C API and C++ together
	NVdecoder* pInstance = reinterpret_cast<NVdecoder*>(pUserData);

	// make sure decoding only happens with a valid object
	// fail if object is invalid
	if(0 == pInstance->m_decoderHandle) return 0;

	// actually decode the frame
	cuErr(cuvidDecodePicture(pInstance->m_decoderHandle, pPicParams));

	pInstance->m_currentDecodeGap++;

	// return value of 1 indicates success
	return 1;
}

// output frames go into a queue
int NVdecoder::output_callback(void *pUserData, CUVIDPARSERDISPINFO* pParDispInfo)
{
	// stitch C API and C++ together
	NVdecoder* pInstance = reinterpret_cast<NVdecoder*>(pUserData);

	// decoded frame info
	CUdeviceptr devPtr = 0;
	unsigned pitch = 0;
	CUVIDPROCPARAMS trashParams;

	// for the queue
	GPUFrame outputFrame;

	// make a GPUFrame object
		// map output
	cuErr(cuvidMapVideoFrame(pInstance->m_decoderHandle, pParDispInfo->picture_index, &devPtr, &pitch, &trashParams));
		// construct GPUFrame
		// height multiplier has to do with NV12 format
	outputFrame = GPUFrame(devPtr, pitch, pInstance->m_width, pInstance->m_height, pInstance->m_width, (pInstance->m_height * 3 / 2), pParDispInfo->timestamp);
		// unmap output
	cuErr(cuvidUnmapVideoFrame(pInstance->m_decoderHandle, devPtr));

	// place into queue
	pInstance->m_outputQueue.push(outputFrame);

	// buffer has one less frame
	pInstance->m_currentDecodeGap--;

	// if stream end set
	if(pInstance->m_eosFlag)
	{
		// buffers empty
		if(0 >= pInstance->m_currentDecodeGap)
		{
			pInstance->m_outputQueue.push(GPUFrame(EOS()));
		}
	}

	// return value of 1 indicates success
	return 1;
}

// static member variables
CUvideoctxlock NVdecoder::s_lock;
bool NVdecoder::s_lockInitialized = false;
int NVdecoder::s_instanceCount = 0;

NVdecoder::NVdecoder(ConcurrentQueue<GPUFrame>& outputQueue): m_outputQueue(outputQueue)
{
	// intialize context state (don't worry about multiple GPUs yet)
	if(false == s_lockInitialized)
	{
		CUdevice mainDevice;
		CUcontext context;
		cuErr(cuInit(0)); // flags argument must be 0
		cuErr(cuDeviceGet(&mainDevice, cudaPrimaryDevice)); // we'll need access to the main device no matter what

		#ifdef TRY_MULTIPLE_GPU

		CUdevice candidateDevice; // try to use this device for decoding
		int deviceCount; // number of installed CUDA devices
		int mainCanAccessCandidate; // C-style bool
		
		cuErr(cuDeviceGetCount(&deviceCount));
		if(1 < deviceCount)
		{
			cuErr(cuDeviceGet(&candidateDevice, cudaSecondaryDevice)); // choose device 1 (device 0 is reserved for performance)
			cuErr(cuDeviceCanAccessPeer(&mainCanAccessCandidate, mainDevice, candidateDevice)); // ensure main GPU can read from candidate
			if(mainCanAccessCandidate) // if peer memory access isn't possible
			{
				// overwrite "main" device so candidate gets used
				// make sure peer access is enabled before starting device 0's access
				mainDevice = candidateDevice;
				std::cout << "using secondary device for decoding" << std::endl;
			}
			else
			{
				std::cerr << "err: can't use P2P memory access" << std::endl;
				std::cerr << "     using primary device for decoding" << std::endl;
			}
		}

		#endif

		cuErr(cuCtxCreate(&context, 0, mainDevice)); // 2nd argument is for flags, none needed

		// need to make context floating for lock
		cuCtxPopCurrent(nullptr);

		// make the lock to hold for all instances
		cuvidCtxLockCreate(&s_lock, context);

		// signal lock creation
		s_lockInitialized = true;
	}

	// create and initialize a params struct to make an nvcuvid parser object
	CUVIDPARSERPARAMS params;

	params.CodecType              = cudaVideoCodec_H264; // magic value
	params.ulMaxNumDecodeSurfaces = DEFAULT_DECODE_SURFACES;
	params.ulClockRate            = DEFAULT_CLOCK_RATE;
	params.ulErrorThreshold       = DEFAULT_ERROR_THRESHOLD;
	params.ulMaxDisplayDelay      = DEFAULT_DECODE_GAP;
	
	memset(params.uReserved1, 0, sizeof(params.uReserved1));

	params.pUserData              = this; // keep track of instances involved
	params.pfnSequenceCallback    = sequence_callback;
	params.pfnDecodePicture       = decode_callback;
	params.pfnDisplayPicture      = output_callback;

	memset(params.pvReserved2, 0, sizeof(params.pvReserved2));

	params.pExtVideoInfo          = NULL; // not currently in use

	// make the aforementioned parser object (CUVIDparser class data member)
	cuErr(cuvidCreateVideoParser(&m_parserHandle, &params));

	s_instanceCount++;
}

NVdecoder::~NVdecoder()
{
	// get rid of cuvid resources (starting to think the underlying library is in C++)
	cuvidDestroyVideoParser(m_parserHandle);
	cuvidDestroyDecoder(m_decoderHandle);

	// get rid of lock when decoders are gone
	s_instanceCount--;
	if(0 >= s_instanceCount && s_lockInitialized)
	{
		cuvidCtxLockDestroy(s_lock);
	}
}

// start processing the coded frame
// cuvidParseVideoData returns before decoding finishes
// must pop from the output queue
int NVdecoder::decodeFrame(const CodedFrame& frame)
{
	CUVIDSOURCEDATAPACKET sdp;

	sdp.flags        = frame.timestamp() & CUVID_PKT_TIMESTAMP; // valid/invalid timestamp
	sdp.payload_size = frame.size();
	sdp.payload      = frame.raw_data();
	sdp.timestamp    = frame.timestamp();

	// parse coded frame and launch sequence, decode, and output
	// callbacks as necessary
	cuErr(cuvidParseVideoData(m_parserHandle, &sdp));

	return 0;
}

// call this after final call to decodeFrame for one contiguous stream of video
void NVdecoder::signalEndOfStream(void)
{
	CUVIDSOURCEDATAPACKET sdp;

	sdp.flags = CUVID_PKT_ENDOFSTREAM;
	sdp.payload_size = 0;
	sdp.payload = nullptr;
	sdp.timestamp = 0;

	// hold on to signal
	m_eosFlag = true;

	// tell cuvid stream is done
	cuErr(cuvidParseVideoData(m_parserHandle, &sdp));
}