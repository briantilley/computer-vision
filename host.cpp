#include <iostream>
#include <iomanip>
#include <thread>
#include "headers/V4L2cam.h"
#include "headers/NVdecoder.h"

using namespace std;

// cuda stuff
#include <cuda_profiler_api.h>

void threadInputDecode(V4L2cam& webcam, NVdecoder& decoder)
{
	// frame from V4L2
	CodedFrame inputFrame;

	while(webcam.isOn())
	{
		cout << "getting frame" << endl;
		// V4L2 blocks until frame comes in
		inputFrame = webcam.retrieveCodedFrame();

		cout << "frame empty: " << inputFrame.empty() << endl;
		cout << "webcam: " << webcam.good() << endl;
		
		// timestamp in frame is valid,
		// output frame will go into queue
		// passed when constructing 'decoder'
		if(!inputFrame.empty())
		{
			cout << "decoding frame" << endl;
			decoder.decodeFrame(inputFrame, CUVID_PKT_TIMESTAMP);
		}
	}

	cout << "end threadInputDecode" << endl;
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

	for(int i = 0; i < 8; ++i) // 3 seconds
	// for(int i = 0; true; ++i) // indefinite runtime
	{
		// thread-safe way to get frame from queue
		cout << "popping frame" << endl;
		decodedQueue.pop(decodedFrame);

		framerate = (int)(1000000 / (decodedFrame.timestamp() - prev_timestamp) + .5f);
		cout << setw(2) << framerate;

		prev_timestamp = decodedFrame.timestamp();

		if(4 == i % 5)
			cout << endl;
		else
			cout << " " << flush;
	}

	// this breaks 'threadInputDecode' from loop
	cout << "turning stream off" << endl;
	webcam.streamOff();

	// wait for all threads to finish
	inputDecodeThread.join();

	cudaProfilerStop();

	return 0;
}
