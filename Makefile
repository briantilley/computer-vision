# compilation files
SRC=host.cpp device.cu V4L2cam.cpp NVdecoder.cpp CudaGLviewer.cpp
OBJ_1=$(SRC:.cpp=.o)
OBJ=$(OBJ_1:.cu=.o)
# OBJ=$(patsubst %.cu,%.o,&(patsubst %.cpp,%.o,$(SRC)))
ALL_H=headers/CudaGLviewer.h headers/constants.h headers/CodedFrame.h headers/GPUFrame.h headers/ConcurrentQueue.h headers/V4L2cam.h headers/NVdecoder.h headers/device.h

# compilers
CPP=g++-5
CUDA=nvcc

# compile for all cuda architectures of installed cards
ARCH=-gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30

all: a.out

host.o: $(ALL_H)

device.o: headers/GPUFrame.h headers/device.h

V4L2cam.o: headers/CodedFrame.h headers/V4L2cam.h

NVdecoder.o: headers/constants.h headers/CodedFrame.h headers/GPUFrame.h headers/ConcurrentQueue.h headers/NVdecoder.h

CudaGLviewer.o: headers/constants.h headers/CudaGLviewer.h headers/GPUFrame.h

%.o: %.cpp
	$(CPP) $< -c --std=gnu++11 -Og

%.o: %.cu
	@# fully optimized
	@# $(CUDA) $< -c --std=c++11 --default-stream per-thread $(ARCH)

	@# not optimized
	@# $(CUDA) $< -c --std=c++11 -Xcicc -O0 -Xptxas -O0 $(ARCH)

	@# use for profiling
	$(CUDA) $< -c --std=c++11 $(ARCH)

a.out: $(OBJ)
	$(CUDA) $^ -o $@ -lnvcuvid -lcuda -lcudart -lGL -lGLEW -lglfw3 -lX11 -lXi -lXxf86vm -lXrandr -lXinerama -lXcursor

clean:
	rm -rf *.o a.out

# run nvprof with less hassle
profile:
	nvprof --profile-from-start off --metrics all ./a.out
