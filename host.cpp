#include <iostream>
#include <iomanip>
#include "headers/V4L2cam.h"

using namespace std;

int main(void)
{
	cout << "Let's go!" << endl;

	unsigned captureWidth, captureHeight;
	V4L2cam webcam(string("/dev/video0"), h264, captureWidth, captureHeight);
	CodedFrame frame;
	unsigned prev_timestamp = 0;

	webcam.streamOn();

	for(int i = 0; i < 90; ++i)
	{
		frame = webcam.retrieveCodedFrame();

		cout << (int)(1000000 / (frame.timestamp() - prev_timestamp) + .5f) << " fps" << flush;
		cout << endl;

		prev_timestamp = frame.timestamp();

		// if(4 == i % 5)
		// 	cout << endl;
		// else
		// 	cout << " " << flush;
	}

	return 0;
}