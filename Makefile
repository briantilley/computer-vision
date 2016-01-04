# compilers
CPP=g++
CUDA=nvcc

# executable name
EXE=a.out
DBG=debug

# compiler options
CPP_OPTIONS=-std=gnu++11 -Og
CUDA_OPTIONS=-std=c++11

# libraries
LINKS=-lnvcuvid -lcuda -lcudart

all: host video decode
	$(CUDA) *.o -o $(EXE) $(CUDA_OPTIONS) $(LINKS)

host: host.cpp
	$(CPP) host.cpp -c $(CPP_OPTIONS)

video: V4L2cam.cpp
	$(CPP) V4L2cam.cpp -c $(CPP_OPTIONS)

decode: NVdecoder.cpp
	$(CPP) NVdecoder.cpp -c $(CPP_OPTIONS)

clean:
	rm -rf *.o $(EXE) $(DBG)
