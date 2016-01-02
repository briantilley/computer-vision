#include "headers/NVdecoder.h"
#include <cstring>

// parser works on basis of callback functions
int sequence_callback(void *pUserData, CUVIDEOFORMAT* pVidFmt)
{
	// stitch C API and C++ together
	NVdecoder* pInstance = reinterpret_cast<NVdecoder*>(pUserData);

	CUVIDDECODECREATEINFO dci;

	// only create the decoder if it doesn't already exist
	if(0 == pInstance->CUdecoder())
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

		dci.vidLock             = pInstance->vidLock(); // come back to this for multithreading
		dci.target_rect.left    = 0;
		dci.target_rect.top     = 0;
		dci.target_rect.right   = dci.ulTargetWidth;
		dci.target_rect.bottom  = dci.ulTargetHeight;

		memset(dci.Reserved2, 0, sizeof(dci.Reserved2));

		// create the decoder
		cuErr(cuvidCreateDecoder(&pInstance->CUdecoder(), &dci));
	}

	// return value of 1 indicates success
	return 1;
}

int decode_callback(void *pUserData, CUVIDPICPARAMS* pPicParams)
{
	// stitch C API and C++ together
	NVdecoder* pInstance = reinterpret_cast<NVdecoder*>(pUserData);

	// make sure decoding only happens with a valid object
	// fail if object is invalid
	if(0 == pInstance->CUdecoder()) return 0;

	// actually decode the frame
	cuErr(cuvidDecodePicture(pInstance->CUdecoder(), pPicParams));

	// return value of 1 indicates success
	return 1;
}

// output frames go into a queue
int output_callback(void *pUserData, CUVIDPARSERDISPINFO* pParDispInfo)
{
	// stitch C API and C++ together
	NVdecoder* pInstance = reinterpret_cast<NVdecoder*>(pUserData);

	// make a GPUFrame object
		// map output
		// copy to cuda C/C++ accessible buffer
		// unmap output
		// construct GPUFrame

	// place into queue
		// push()

	return 0;
}

// static member variables
CUvideoctxlock NVdecoder::s_lock;
bool NVdecoder::s_lockInitialized = false;
int NVdecoder::s_instanceCount = 0;

NVdecoder::NVdecoder(ConcurrentQueue<GPUFrame>& outputQueue)
{
	// location for output frames
	m_outputQueue = outputQueue;

	// intialize context state (don't worry about multiple GPUs yet)
	if(false == s_lockInitialized)
	{
		CUdevice device;
		CUcontext context;
		cuErr(cuInit(0)); // flags argument must be 0
		cuErr(cuDeviceGet(&device, 0)); // 2nd argument is device number
		cuErr(cuCtxCreate(&context, 0, device)); // 2nd argument is for flags, none needed

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
int NVdecoder::decodeFrame(const CodedFrame& frame, CUvideopacketflags flags)
{
	CUVIDSOURCEDATAPACKET sdp;

	sdp.flags        = flags;
	sdp.payload_size = frame.size();
	sdp.payload      = frame.raw_data();
	sdp.timestamp    = frame.timestamp();

	// parse coded frame and launch sequence, decode, and output
	// callbacks as necessary
	cuErr(cuvidParseVideoData(CUparser(), &sdp));

	return 0;
}
