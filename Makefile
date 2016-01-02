# compilers
CPP=g++
CUDA=nvcc

# executable name
EXE=a.out
DBG=debug

# compiler options
CPP_OPTS=-std=gnu++11
CUDA_OPTS=-std=c++11

# libraries
LINKS=-lnvcuvid -lcuda

all: host video decode
	$(CUDA) *.o -o $(EXE) $(CUDA_OPTS) $(LINKS)

host: host.cpp
	$(CPP) host.cpp -c $(CPP_OPTS)

video: V4L2cam.cpp
	$(CPP) V4L2cam.cpp -c $(CPP_OPTS)

decode: NVdecoder.cpp GPUFrame.cpp
	$(CPP) NVdecoder.cpp -c $(CPP_OPTS)
	$(CPP) GPUFrame.cpp -c $(CPP_OPTS)

clean:
	rm -rf *.o $(EXE) $(DBG)
