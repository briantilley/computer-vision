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
		CUvideodecoder tempDecoder;
		cuErr(cuvidCreateDecoder(&tempDecoder, &dci));
		pInstance->setCUdecoder(tempDecoder);
	}

	// fill width and height of decoder
	pInstance->setVideoWidth(pVidFmt->display_area.right);
	pInstance->setVideoHeight(pVidFmt->display_area.bottom);

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

	pInstance->incrementDecodeGap();

	// return value of 1 indicates success
	return 1;
}

// output frames go into a queue
int output_callback(void *pUserData, CUVIDPARSERDISPINFO* pParDispInfo)
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
	cuErr(cuvidMapVideoFrame(pInstance->CUdecoder(), pParDispInfo->picture_index, &devPtr, &pitch, &trashParams));
		// construct GPUFrame
		// height multiplier has to do with NV12 format
	outputFrame = GPUFrame(devPtr, pitch, pInstance->videoWidth(), pInstance->videoHeight(), pInstance->videoWidth(), (pInstance->videoHeight() * 3 / 2), pParDispInfo->timestamp);
		// unmap output
	cuErr(cuvidUnmapVideoFrame(pInstance->CUdecoder(), devPtr));

	// place into queue
	pInstance->pushFrame(outputFrame);

	// buffer has one less frame
	pInstance->decrementDecodeGap();

	// if stream end set
	if(pInstance->eosFlag())
	{
		// buffers empty
		if(0 >= pInstance->decodeGap())
		{
			pInstance->pushFrame(GPUFrame(EOS()));
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