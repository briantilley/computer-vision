#ifndef GPU_FRAME_H
#define GPU_FRAME_H

// hopefully mixing driver and runtime won't wreak havoc
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>

// development
#include <iostream>

// error checking for cuda runtime calls
#define cudaErr(err) cudaError(err, __FILE__, __LINE__)
inline void cudaError(cudaError_t err, const char file[], uint32_t line)
{
	if(cudaSuccess != err)
	{
		std::cerr << "[" << file << ":" << line << "] ";
		std::cerr << cudaGetErrorName(err) << std::endl;
		exit(err);
	}
}

// same as above without exceptional behavior
#define cudaErrNE(err) cudaErrorNE(err, __FILE__, __LINE__)
inline void cudaErrorNE(cudaError_t err, const char file[], uint32_t line)
{
	if(cudaSuccess != err)
	{
		std::cerr << "[" << file << ":" << line << "] ";
		std::cerr << cudaGetErrorName(err) << std::endl;
	}
}

// helper class to signal end of stream
class EOS
{
public:
	EOS() = default;
	~EOS() = default;
};

// client code is trusted not to modify the contents of the frame
class GPUFrame
{
private:

	std::shared_ptr<void> m_deviceData;
	size_t m_pitch;
	unsigned m_width;
	unsigned m_height;
	unsigned m_timestamp; // time value in microseconds (absolute value is arbitrary)
	bool m_endOfStream; // signifies last frame in the stream

public:

	// let all members remain in the empty state
	GPUFrame(): m_pitch(0), m_width(0), m_height(0), m_timestamp(0), m_endOfStream(false) { }

	// make an entirely new allocation
	// creating multiple instances from the same pointer = undefined behavior
	GPUFrame(unsigned imageWidth, unsigned imageHeight, unsigned allocationCols, unsigned allocationRows,
			 unsigned timestamp, bool eos=false): m_pitch(0), m_width(imageWidth), m_height(imageHeight), m_timestamp(timestamp), m_endOfStream(eos)
	{
		// get space from CUDA
		void* newAllocation;
		cudaErr(cudaMallocPitch(&newAllocation, &m_pitch, static_cast<size_t>(allocationCols), static_cast<size_t>(allocationRows)));

		// track allocation with the shared_ptr
		m_deviceData = std::shared_ptr<void>(newAllocation, [=](void* p){ cudaErrNE(cudaFree(p)); });
	}

	// copy from given location
	GPUFrame(CUdeviceptr devPtr, unsigned pitch,
			 unsigned imageWidth, unsigned imageHeight, unsigned allocationCols, unsigned allocationRows,
			 unsigned timestamp, bool eos=false): GPUFrame(imageWidth, imageHeight, allocationCols, allocationRows, timestamp)
	{
		// copy into a more permanent chunk of memory allocated by above ctor
		cudaErr(cudaMemcpy2D(data(), m_pitch, reinterpret_cast<void*>(devPtr), pitch,
							allocationCols, allocationRows, cudaMemcpyDeviceToDevice));
	}

	GPUFrame(EOS eos): m_pitch(0), m_width(0), m_height(0), m_timestamp(0), m_endOfStream(true) { }

	// let C++ copy all member data
	GPUFrame(const GPUFrame&) = default;
	GPUFrame& operator=(const GPUFrame&) = default;

	~GPUFrame() = default;

	// check if frame is empty (possible use: error signaling)
	bool empty(void) const { return !(static_cast<bool>(m_width) && static_cast<bool>(m_height)); }

	// check if this is the last frame to be recieved
	bool eos(void) const { return m_endOfStream; }

	void* data(void) { return m_deviceData.get(); }; // pointer this is wrapped around
	size_t pitch(void) const { return /*static_cast<unsigned>*/(m_pitch); } // number of bytes between the start of one row and the next
	unsigned width(void) const { return m_width; } // dimensions (in pixels) of the image
	unsigned height(void) const { return m_height; } // ^
	unsigned timestamp(void) const { return	m_timestamp; }; // capture time of frame

	// time between frames
	int operator-(const GPUFrame& right) const { return m_timestamp - right.m_timestamp; }
};

#endif
