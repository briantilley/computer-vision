#include <iostream>
#include <iomanip>
#include <thread>
#include "headers/V4L2cam.h"
#include "headers/NVdecoder.h"
#include "headers/device.h"

#include <unistd.h>

using namespace std;

// cuda stuff
#include <cuda_profiler_api.h>

#define FRAMES_TO_PROCESS 300

void threadInputDecode(V4L2cam& webcam, NVdecoder& decoder)
{
	// frame from V4L2
	CodedFrame inputFrame;
	unsigned retrievedCount = 0;

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

		if(FRAMES_TO_PROCESS <= retrievedCount)
			webcam.streamOff();
	}
	cout << endl;

	// if webcam is off, stream is over
	// and decoding buffer needs to be flushed
	// (also pushes end of stream frame to output queue)
	decoder.signalEndOfStream();
}

// takes decoded input from the queue and decodes with CUDA functions
void threadPostProcess(ConcurrentQueue<GPUFrame>& inputQueue)
{
	// frame popped from queue
	GPUFrame NV12input;

	while(true) // break upon receiving end of stream frame
	{
		// thread-safe input from decoder
		inputQueue.pop(NV12input);

		if(!NV12input.eos()) // if stream isn't finished
		{
			// convert the frame and let it go to waste
			if(!NV12input.empty())
			{
				NV12toRGB(NV12input);
			}
			else
				cout << "empty frame from decoder" << flush;
		}
		else
		{
			break;
		}
	}
	cout << endl;
}

int main(void)
{
	cout << "Let's go!" << endl;
	cout << std::thread::hardware_concurrency() << " concurrent threads supported" << endl;
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
	ConcurrentQueue<GPUFrame> decodedQueue;

	// decoder
	NVdecoder gpuDecoder(decodedQueue);

	// begin
	webcam.streamOn();

	// hand input/decode over to a new thread
	inputDecodeThread = std::thread(threadInputDecode, std::ref(webcam), std::ref(gpuDecoder));

	// start the post-processing thread
	// cudaProfilerStart();
	postProcessThread = std::thread(threadPostProcess, std::ref(decodedQueue));

	// // give time for threads to work a bit
	// sleep(2);

	// // end of stream breaks threads from loop
	// webcam.streamOff();

	// wait for all threads to finish
	inputDecodeThread.join();
	postProcessThread.join();
	// cudaProfilerStop();

	return 0;
}
