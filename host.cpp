#include <iostream>
#include <iomanip>
#include "headers/V4L2cam.h"
#include "headers/NVdecoder.h"

using namespace std;

int main(void)
{
	cout << "Let's go!" << endl;

	unsigned captureWidth = 1920, captureHeight = 1080;
	V4L2cam webcam(string("/dev/video0"), h264, captureWidth, captureHeight);
	CodedFrame frame;
	NVdecoder gpuDecoder;
	unsigned prev_timestamp = 0;

	webcam.streamOn();

	for(int i = 0; i < 300; ++i)
	{
		frame = webcam.retrieveCodedFrame();
		gpuDecoder.decodeFrame(frame);

		if(2 == i % 3)
		{
			cout << (int)(3000000 / (frame.timestamp() - prev_timestamp) + .5f) << " fps" << flush;
			cout << endl;

			prev_timestamp = frame.timestamp();
		}

		// if(4 == i % 5)
		// 	cout << endl;
		// else
		// 	cout << " " << flush;
	}

	return 0;
}