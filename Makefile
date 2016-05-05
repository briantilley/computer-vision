# compiler
CPP=g++

DRIVER_HEADERS=headers/V4L2cam.h headers/RawFrame.h headers/RawFrameCPU.h headers/Decoder.h headers/DecoderCPU.h
V4L2_CAM_HEADERS=headers/CodedFrame.h headers/V4L2cam.h
DECODER_CPU_HEADERS=headers/Decoder.h headers/DecoderCPU.h headers/RawFrame.h headers/RawFrameCPU.h headers/CodedFrame.h

all: driver.o V4L2cam.o DecoderCPU.o
	$(CPP) $^ -o a.out --std=gnu++11 -lavcodec -lavutil -lswresample -lpthread -lz

V4L2cam.o: $(V4L2_CAM_HEADERS)

driver.o: $(DRIVER_HEADERS)

DecoderCPU.o: $(DECODER_CPU_HEADERS)

%.o: %.cpp
	$(CPP) $< -c --std=gnu++11 -Og

clean:
	rm -rf *.o a.out