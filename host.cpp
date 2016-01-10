#include <iostream>
#include <iomanip>
#include <thread>
#include "headers/V4L2cam.h"
#include "headers/NVdecoder.h"
#include "headers/CudaGLviewer.h"
#include "headers/device.h"
#include "headers/constants.h"

#include <unistd.h>

using namespace std;

// cuda stuff
#include <cuda_profiler_api.h>

unsigned gFramesToProcess = DEFAULT_FRAMES_TO_PROCESS;
unsigned cudaPrimaryDevice = 0;
unsigned cudaSecondaryDevice = 1;

void threadInputDecode(V4L2cam& webcam, NVdecoder& decoder)
{
	// frame from V4L2
	CodedFrame inputFrame;
	unsigned retrievedCount = 0;

	// make sure we're crunching numbers on the fastest GPU
	// every thread needs to call this to use the same GPU
	cudaErr(cudaSetDevice(cudaPrimaryDevice));

	while(webcam.isOn())
	{
		// V4L2 blocks until frame comes in
		inputFrame = webcam.retrieveCodedFrame();

		// timestamp in frame is valid,
		// output frame will go into queue
		// passed when constructing 'decoder'
		if(!inputFrame.empty())
		{
			cout << "." << flush;
			retrievedCount++;
			decoder.decodeFrame(inputFrame);
		}

		if(gFramesToProcess <= retrievedCount)
			webcam.streamOff();
	}
	cout << endl;

	// if webcam is off, stream is over
	// and decoding buffer needs to be flushed
	// (also pushes end of stream frame to output queue)
	decoder.signalEndOfStream();
}

// takes decoded input from the queue and decodes with CUDA functions
void threadPostProcess(ConcurrentQueue<GPUFrame>& inputQueue, ConcurrentQueue<GPUFrame>& displayQueue)
{
	// frame popped from queue
	GPUFrame NV12input, RGBAframe;

	// make sure we're crunching numbers on the fastest GPU
	// every thread needs to call this to use the same GPU
	cudaErr(cudaSetDevice(cudaPrimaryDevice));

	while(true) // break upon receiving end of stream frame
	{
		// thread-safe input from decoder
		inputQueue.pop(NV12input);

		if(!NV12input.eos()) // if stream isn't finished
		{
			// convert the frame and let it go to waste
			if(!NV12input.empty())
			{
				RGBAframe = NV12toRGBA(NV12input);
			}
			else
				cout << "empty frame from decoder" << flush;
		}
		else
		{
			// pass EOS signal along
			displayQueue.push(EOS());
			break;
		}
	}
	cout << endl;
}

int main(int argc, char* argv[])
{
	// arguments
	if(argc == 2) // first = number of frames
		gFramesToProcess = atoi(argv[1]);
	else if(argc == 3) // second = CUDA card to use
	{
		cudaPrimaryDevice = atoi(argv[2]) ? 1 : 0;
		cudaSecondaryDevice = atoi(argv[2]) ? 0 : 1;
	}

	// identify GPUs
	cudaDeviceProp properties;

	cudaErr(cudaGetDeviceProperties(&properties, cudaPrimaryDevice));
	cout << "primary CUDA device: " << properties.name << endl;

	cudaErr(cudaGetDeviceProperties(&properties, cudaSecondaryDevice));
	cout << "secondary CUDA device: " << properties.name << endl;

	cout << endl;


	// development
	GPUFrame decodedFrame;

	// metrics
	float framerate = 0;
	unsigned prev_timestamp = 0;
	float framerateAccumulator = 0;
	int frameCount = 0;

	// input/decode thread
	std::thread inputDecodeThread, postProcessThread;

	// camera
	unsigned captureWidth = 1920, captureHeight = 1080;
	V4L2cam webcam(string("/dev/video0"), h264, captureWidth, captureHeight);

	// decoder output queue
	ConcurrentQueue<GPUFrame> decodedQueue, displayQueue;

	// decoder
	NVdecoder gpuDecoder(decodedQueue);

	// make sure we're crunching numbers on the fastest GPU
	// every thread needs to call this to use the same GPU
	cudaErr(cudaSetDevice(cudaPrimaryDevice));

	CudaGLviewer::initGlobalState();
	CudaGLviewer viewer(1920, 1080, "roses");

	// begin
	webcam.streamOn();

	// hand input/decode over to a new thread
	inputDecodeThread = std::thread(threadInputDecode, std::ref(webcam), std::ref(gpuDecoder));

	// start the post-processing thread
	cudaProfilerStart();
	postProcessThread = std::thread(threadPostProcess, std::ref(decodedQueue), std::ref(displayQueue));

	// // give time for threads to work a bit
	// sleep(2);

	// main thread
	GPUFrame displayFrame;
	while(!displayFrame.eos())
	{
		displayQueue.pop(displayFrame);

		if(!displayFrame.eos())
			viewer.drawFrame(displayFrame);
	}

	// // end of stream breaks threads from loop
	// webcam.streamOff();

	// wait for all threads to finish
	inputDecodeThread.join();
	postProcessThread.join();
	cudaProfilerStop();

	return 0;
}
