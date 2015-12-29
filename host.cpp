#include <iostream>
#include "headers/V4L2cam.h"

using namespace std;

int main(void)
{
	cout << "Let's go!" << endl;

	unsigned captureWidth, captureHeight;
	V4L2cam webcam(string("/dev/video0"), h264, captureWidth, captureHeight);
	CodedFrame frame;

	webcam.streamOn();

	for(int i = 0; i < 90; ++i)
	{
		frame = webcam.retrieveCodedFrame();

		cout << frame.size() << "/" << static_cast<short>(frame.raw_data()[frame.size() - 1]);

		if(4 == i % 5)
			cout << endl;
		else
			cout << " " << flush;
	}

	return 0;
}