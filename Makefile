# compiler
CPP=g++

all: driver.o V4L2cam.o
	$(CPP) $^ -o a.out --std=gnu++11

V4L2cam.o: headers/CodedFrame.h headers/V4L2cam.h

driver.o: headers/RawFrame.h headers/RawFrameCPU.h

%.o: %.cpp
	$(CPP) $< -c --std=gnu++11 -Og

clean:
	rm -rf *.o a.out