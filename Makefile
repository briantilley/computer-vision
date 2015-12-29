# compilers
CPP=g++
CUDA=nvcc

# executable name
EXE=a.out
DBG=debug

# compiler options
CPP_OPTS=-std=gnu++11
CUDA_OPTS=-std=c++11

all: host video
	$(CUDA) *.o -o $(EXE) $(CUDA_OPTS)

host: host.cpp
	$(CPP) host.cpp -c $(CPP_OPTS)

video: V4L2cam.cpp
	$(CPP) V4L2cam.cpp -c $(CPP_OPTS)
	$(CPP) CodedFrame.cpp -c $(CPP_OPTS)

clean:
	rm -rf *.o $(EXE) $(DBG)