# compilation files
SRC=host.cpp device.cu V4L2cam.cpp NVdecoder.cpp
OBJ_1=$(SRC:.cpp=.o)
OBJ=$(OBJ_1:.cu=.o)
# OBJ=$(patsubst %.cu,%.o,&(patsubst %.cpp,%.o,$(SRC)))
ALL_H=headers/CodedFrame.h headers/GPUFrame.h headers/ConcurrentQueue.h headers/V4L2cam.h headers/NVdecoder.h headers/device.h

# compilers
CPP=g++-5
CUDA=nvcc

all: a.out

host.o: $(ALL_H)

V4L2cam.o: headers/CodedFrame.h

NVdecoder.o: headers/CodedFrame.h headers/GPUFrame.h headers/ConcurrentQueue.h

%.o: %.cpp
	$(CPP) $< -c --std=gnu++11 -Og

%.o: %.cu
	$(CUDA) $< -c --std=c++11 --default-stream per-thread -arch sm_30

a.out: $(OBJ)
	$(CUDA) $^ -o $@ -lnvcuvid -lcuda -lcudart

clean:
	rm -rf *.o a.out

# run nvprof with less hassle
profile:
	nvprof ./a.out --profile-from-start off --metrics gld_efficiency
