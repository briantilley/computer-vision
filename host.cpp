#include <iostream>
#include <iomanip>
#include <thread>
#include "headers/V4L2cam.h"
#include "headers/NVdecoder.h"

#include <unistd.h>

using namespace std;

// cuda stuff
// #include <cuda_profiler_api.h>

void threadInputDecode(V4L2cam& webcam, NVdecoder& decoder)
{
	// frame from V4L2
	CodedFrame inputFrame;

	while(webcam.isOn())
	{
		// V4L2 blocks until frame comes in
		inputFrame = webcam.retrieveCodedFrame();

		// timestamp in frame is valid,
		// output frame will go into queue
		// passed when constructing 'decoder'
		if(!inputFrame.empty())
		{
			decoder.decodeFrame(inputFrame);
		}
	}

	// if webcam is off, stream is over
	// and decoding buffer needs to be flushed
	decoder.signalEndOfStream();
}

// takes decoded input from the queue and decodes with CUDA functions
void threadPostProcess(ConcurrentQueue<GPUFrame> inputQueue)
{

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
	std::thread inputDecodeThread;

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

	for(; frameCount < 90; ++frameCount) // 1 second
	// for(int i = 0; true; ++i) // indefinite runtime
	{
		// thread-safe way to get frame from queue
		decodedQueue.pop(decodedFrame);

		// framerate business
		framerate = 1000000.f / (decodedFrame.timestamp() - prev_timestamp);
		framerateAccumulator += framerate;
		prev_timestamp = decodedFrame.timestamp();
	}

	// this breaks 'threadInputDecode' from loop
	webcam.streamOff();

	// wait for all threads to finish
	inputDecodeThread.join();

	for(; !(gpuDecoder.empty() && decodedQueue.empty()); ++frameCount) // pop frames until both are empty
	{
		// thread-safe way to get frame from queue
		decodedQueue.pop(decodedFrame);

		// framerate business
		framerate = 1000000.f / (decodedFrame.timestamp() - prev_timestamp);
		framerateAccumulator += framerate;
		prev_timestamp = decodedFrame.timestamp();
	}

	cout << frameCount << " frames" << endl;
	cout << "average framerate: " << framerateAccumulator / frameCount << " fps" << endl;

	return 0;
}
