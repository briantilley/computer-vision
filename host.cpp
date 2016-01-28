#include <iostream>
#include <iomanip>
#include <thread>
#include "headers/V4L2cam.h"
#include "headers/NVdecoder.h"
#include "headers/CudaGLviewer.h"
#include "headers/device.h"
#include "headers/constants.h"

#include <unistd.h>

#define CUDA_PROFILING
#define GL_VIEWER_UPDATE_INTERVAL 1000
#define VIDEO_WIDTH 160
#define VIDEO_HEIGHT 120

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
			// cout << decoder.decodeGap() << " " << flush;
			retrievedCount++;
			decoder.decodeFrame(inputFrame);
		}

		// if(gFramesToProcess <= retrievedCount)
		// 	webcam.streamOff();
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
			// convert the frame
			if(!NV12input.empty())
			{
				RGBAframe = NV12toRGBA(NV12input);
				displayQueue.push(RGBAframe);
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

void threadDisplay(CudaGLviewer& viewer, ConcurrentQueue<GPUFrame>& displayQueue)
{
	GPUFrame displayFrame;

	while(true)
	{
		displayQueue.pop(displayFrame);

		if(!displayFrame.eos() && viewer)
		{
			viewer.drawFrame(displayFrame);
		}
		else // end of stream/invalidated viewer object
		{
			break;
		}
	}
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

	// input/decode thread
	std::thread inputDecodeThread, postProcessThread, displayThread;

	// camera
	unsigned captureWidth = VIDEO_WIDTH, captureHeight = VIDEO_HEIGHT;
	V4L2cam webcam(string("/dev/video0"), h264, captureWidth, captureHeight);

	if(!webcam)
		exit(EXIT_FAILURE);

	// frame queues between threads
	ConcurrentQueue<GPUFrame> decodedQueue, displayQueue;

	// decoder
	NVdecoder gpuDecoder(decodedQueue);

	// begin
	webcam.streamOn();

	// hand input/decode over to a new thread
	inputDecodeThread = std::thread(threadInputDecode, std::ref(webcam), std::ref(gpuDecoder));

	#ifdef CUDA_PROFILING
		// start the post-processing thread
		cudaProfilerStart();
	#endif

	postProcessThread = std::thread(threadPostProcess, std::ref(decodedQueue), std::ref(displayQueue));

	// display/input
	CudaGLviewer::initGlobalState();
	ConcurrentQueue<KeyEvent> keyInputQueue;
	KeyEvent currentEvent;
	CudaGLviewer viewer(captureWidth, captureHeight, "input", &keyInputQueue);

	if(!viewer)
		exit(EXIT_FAILURE);

	displayThread = std::thread(threadDisplay, std::ref(viewer), std::ref(displayQueue));
	bool autofocus;
	while(webcam.isOn() && viewer)
	{
		CudaGLviewer::update();

		// input handling
		while(!keyInputQueue.empty())
		{
			// get the key event
			keyInputQueue.pop(currentEvent);

			if(currentEvent.action != ACTION_RELEASE)
			{
				switch(currentEvent.key)
				{
					case KEY_ARROW_UP:
						webcam.changeControl(EXPOSURE, 5);
					break;

					case KEY_ARROW_DOWN:
						webcam.changeControl(EXPOSURE, -5);
					break;

					case KEY_F:
						autofocus = webcam.getControl(AUTOFOCUS);
						webcam.setControl(AUTOFOCUS, !autofocus);
						cout << endl << "autofocus " << (!autofocus ? "on" : "off") << endl;
					break;

					case KEY_R:
						webcam.changeControl(FOCUS, 5);
					break;

					case KEY_V:
						webcam.changeControl(FOCUS, -5);
					break;

				}
			}
		}

		usleep(GL_VIEWER_UPDATE_INTERVAL);
	}

	// triggers all other threads to stop
	webcam.streamOff();

	// causes some XIO error
	// CudaGLviewer::destroyGlobalState();

	// wait for all threads to finish
	inputDecodeThread.join();
	postProcessThread.join();

	displayThread.join();
	
	#ifdef CUDA_PROFILING
		cudaProfilerStop();
	#endif

	return 0;
}
