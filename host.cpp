#include <iostream>
#include <iomanip>
#include <thread>
#include "headers/V4L2cam.h"
#include "headers/NVdecoder.h"

#include <unistd.h>

using namespace std;

// cuda stuff
#include <cuda_profiler_api.h>

void threadInputDecode(V4L2cam& webcam, NVdecoder& decoder)
{
	// frame from V4L2
	CodedFrame inputFrame;

	int i;
	for(i = 0; webcam.isOn(); ++i)
	{
		// V4L2 blocks until frame comes in
		inputFrame = webcam.retrieveCodedFrame();

		// timestamp in frame is valid,
		// output frame will go into queue
		// passed when constructing 'decoder'
		if(!inputFrame.empty())
		{
			decoder.decodeFrame(inputFrame, CUVID_PKT_TIMESTAMP);
		}
	}

	cout << i << " frames retrieved" << endl;
}

// takes decoded input from the queue and decodes with CUDA functions
void threadPostProcess(ConcurrentQueue<GPUFrame> inputQueue)
{

}

int main(void)
{
	cout << "Let's go!" << endl;
	cout << std::thread::hardware_concurrency() << " concurrent threads supported" << endl;

	// development
	GPUFrame decodedFrame;
	int framerate = 0;
	unsigned prev_timestamp = 0;
	int framerateAccumulator = 0;
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

	inputDecodeThread = std::thread(threadInputDecode, std::ref(webcam), std::ref(gpuDecoder));

	// sleep(1);

	for(int i = 0; i < 30; ++i) // 1 second
	// for(int i = 0; true; ++i) // indefinite runtime
	{
		// thread-safe way to get frame from queue
		decodedQueue.pop(decodedFrame);

		framerate = (int)(1000000 / (decodedFrame.timestamp() - prev_timestamp) + .5f);
		framerateAccumulator += framerate;
		frameCount++;
		cout << setw(2) << framerate << "-" << gpuDecoder.decodeGap();

		prev_timestamp = decodedFrame.timestamp();

		if(4 == i % 5)
			cout << endl;
		else
			cout << " " << flush;
	}

	// this breaks 'threadInputDecode' from loop
	webcam.streamOff();

	// wait for all threads to finish
	inputDecodeThread.join();

	// gpuDecoder.signalEndOfStream();

	cout << "remaining frames" << endl;
	int i;
	for(i = 0; !gpuDecoder.empty(); ++i)
	{
		// thread-safe way to get frame from queue
		cout << "decode gap: " << gpuDecoder.decodeGap() << endl;
		decodedQueue.pop(decodedFrame);

		framerate = (int)(1000000 / (decodedFrame.timestamp() - prev_timestamp) + .5f);
		framerateAccumulator += framerate;
		frameCount++;
		cout << setw(2) << framerate << "-" << gpuDecoder.decodeGap();

		prev_timestamp = decodedFrame.timestamp();

		if(4 == i % 5)
			cout << endl;
		else
			cout << " " << flush;
	}
	if(4 != i % 5) cout << endl;

	cout << "average framerate: " << framerateAccumulator / frameCount << endl;

	cudaProfilerStop();

	return 0;
}
