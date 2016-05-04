#ifndef GPU_FRAME_H
#define GPU_FRAME_H

// hopefully mixing driver and runtime won't wreak havoc
#include <cuda.h>
#include <cuda_runtime.h>

// development
#include <iostream>

#include "RawFrame.h"

// error checking for cuda runtime calls (memory functions)
#define cudaErr(err) cudaError(err, __FILE__, __LINE__)
inline void cudaError(cudaError_t err, const char file[], uint32_t line)
{
	if(cudaSuccess != err)
	{
		std::cerr << "[" << file << ":" << line << "] ";
		std::cerr << cudaGetErrorName(err) << std::endl;
	}
}

// client code is trusted not to modify the contents of the frame
class RawFrameGPU: public RawFrame
{
public:

	// let all members remain in the empty state
	RawFrameGPU(): m_pitch(0), m_width(0), m_height(0), m_timestamp(0), m_endOfStream(false) { }

	RawFrameGPU(bool eos): m_pitch(0), m_width(0), m_height(0), m_timestamp(0), m_endOfStream(eos) { }

	// make an entirely new allocation
	RawFrameGPU(unsigned imageWidth, unsigned imageHeight, unsigned allocationCols, unsigned allocationRows,
			 unsigned timestamp, bool eos=false): m_pitch(0), m_width(imageWidth), m_height(imageHeight), m_timestamp(timestamp), m_endOfStream(eos)
	{
		// get space from CUDA
		void* newAllocation;
		cudaErr(cudaMallocPitch(&newAllocation, &m_pitch, static_cast<size_t>(allocationCols), static_cast<size_t>(allocationRows)));

		// track allocation with the shared_ptr
		m_deviceData = std::shared_ptr<void>(newAllocation, [=](void* p){ cudaErr(cudaFree(p)); });
	}

	// copy from given location
	RawFrameGPU(CUdeviceptr devPtr, unsigned pitch,
			 unsigned imageWidth, unsigned imageHeight, unsigned allocationCols, unsigned allocationRows,
			 unsigned timestamp, bool eos=false): RawFrameGPU(imageWidth, imageHeight, allocationCols, allocationRows, timestamp)
	{
		// copy into a more permanent chunk of memory allocated by above ctor
		cudaErr(cudaMemcpy2D(data(), m_pitch, reinterpret_cast<void*>(devPtr), pitch,
							allocationCols, allocationRows, cudaMemcpyDeviceToDevice));
	}

	// let C++ copy all member data
	RawFrameGPU(const RawFrameGPU&) = default;
	RawFrameGPU& operator=(const RawFrameGPU&) = default;

	~RawFrameGPU() = default;

	// time between frames
	int operator-(const RawFrameGPU& right) const { return m_timestamp - right.m_timestamp; }
};

#endif
