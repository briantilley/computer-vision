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
inline void cudaError(cudaError_t err, const char file[], uint32_t line, bool abort=true)
{
	if(cudaSuccess != err)
	{
		std::cerr << "[" << file << ":" << line << "] ";
		std::cerr << cudaGetErrorName(err) << std::endl;
		if(abort) exit(err);
	}
}

// client code is trusted not to modify the contents of the frame
class GPUFrame
{
private:

	std::shared_ptr<void> m_deviceData;
	unsigned m_pitch = 0;
	unsigned m_width = 0;
	unsigned m_height = 0;
	unsigned m_timestamp = 0; // time value in microseconds (absolute value is arbitrary)

public:

	GPUFrame() = default; // let everything remain in the empty state

	GPUFrame(CUdeviceptr devPtr, unsigned pitch, unsigned width, unsigned height, unsigned timestamp): m_width(width), m_height(height), m_timestamp(timestamp)
	{
		// copy into a more permanent chunk of memory
		void* devPtr_copy;
		cudaErr(cudaMallocPitch(&devPtr_copy, reinterpret_cast<size_t*>(&m_pitch), static_cast<size_t>(width), static_cast<size_t>(height)));
		cudaErr(cudaMemcpy2D(devPtr_copy, m_pitch, reinterpret_cast<void*>(devPtr), pitch,
							width, height, cudaMemcpyDeviceToDevice));

		// update the shared pointer
		m_deviceData = std::shared_ptr<void>(devPtr_copy, cudaFree);
	}

	GPUFrame(const GPUFrame& toCopy)
	{
		m_deviceData = toCopy.m_deviceData;
		m_pitch = toCopy.m_pitch;
		m_width = toCopy.m_width;
		m_height = toCopy.m_height;
		m_timestamp = toCopy.m_timestamp;
	}

	void operator=(const GPUFrame& right)
	{
		m_deviceData = right.m_deviceData;
		m_pitch = right.m_pitch;
		m_width = right.m_width;
		m_height = right.m_height;
		m_timestamp = right.m_timestamp;
	}

	~GPUFrame() = default;

	// check if frame is empty (possible use: error signaling)
	bool empty() const { return static_cast<bool>(m_width) && static_cast<bool>(m_height); }

	void* data() const { m_deviceData.get(); }; // pointer this is wrapped around
	unsigned timestamp() const { return	m_timestamp; }; // capture time of frame
};

#endif
