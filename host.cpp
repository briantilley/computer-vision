#include <iostream>
#include <iomanip>
#include "headers/V4L2cam.h"
#include "headers/NVdecoder.h"

using namespace std;

int main(void)
{
	// development
	cout << "Let's go!" << endl;
	int framerate = 0;
	CodedFrame frame;
	unsigned prev_timestamp = 0;

	// camera
	unsigned captureWidth = 1920, captureHeight = 1080;
	V4L2cam webcam(string("/dev/video0"), h264, captureWidth, captureHeight);

	// decoder output queue
	ConcurrentQueue<GPUFrame> decodedQueue;

	// decoder
	NVdecoder gpuDecoder(decodedQueue);

	// begin
	webcam.streamOn();

	for(int i = 0; i < 90; ++i) // 3 seconds
	// for(int i = 0; true; ++i) // indefinite runtime
	{
		frame = webcam.retrieveCodedFrame();
		// tell the decoder timestamp is valid
		gpuDecoder.decodeFrame(frame, CUVID_PKT_TIMESTAMP);

		framerate = (int)(1000000 / (frame.timestamp() - prev_timestamp) + .5f);
		cout << setw(2) << framerate;

		prev_timestamp = frame.timestamp();

		if(4 == i % 5)
			cout << endl;
		else
			cout << " " << flush;
	}

	return 0;
}
