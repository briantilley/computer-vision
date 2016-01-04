#include <cuda_runtime.h>
#include "headers/device.h"

// development
#include <iostream>
#include <cuda_profiler_api.h>

// use this when grid size perfectly matches image size (each thread is 4 pixels)
__global__
void kernelNV12toRGBA(const void* const input, const unsigned pitchInput,
					  void* const output, const unsigned pitchOutput)
{
	// dimensions of the grid
	// (# of blocks) * (threads per block)
	const unsigned gridWidth = gridDim.x * blockDim.x;
	const unsigned gridHeight = gridDim.y * blockDim.y;

	// position within the grid
	// (threads per block) * (position of block in grid) + (position of thread in block)
	const unsigned gridXidx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned gridYidx = blockDim.y * blockIdx.y + threadIdx.y;

	// NV12 global reads for 8 pixels
	// address calculation from inner to outer access
	// convert to byte array, position to proper row,
	// convert to row array, position to proper column
	word packedYbytes = reinterpret_cast<const word*>(reinterpret_cast<const byte*>(input) + gridYidx * pitchInput)[gridXidx];
	word packedUVbytes = reinterpret_cast<const word*>(reinterpret_cast<const byte*>(input) + (gridHeight + gridYidx) * pitchInput)[gridXidx];
}

// use this when grid has threads outside the image
__global__
void paddedKernelNV12toRGBA(const word* const input, const unsigned pitchInput,
							word* const output, const unsigned pitchOutput,
							const unsigned pixelsWidth, const unsigned pixelsHeight)
{

}

// when this works, modify it to push to a ConcurrentQueue<GPUFrame>
GPUFrame NV12toRGB(GPUFrame& NV12frame, const bool makeAlpha)
{
	GPUFrame outputFrame;

	// make an object for the output image
	unsigned allocationRows = (makeAlpha ? 4 : 3) * NV12frame.height();
	unsigned allocationCols = NV12frame.width();

	outputFrame = GPUFrame(NV12frame.width(), NV12frame.height(), allocationCols, allocationRows, NV12frame.timestamp());

	if(0 == NV12frame.width() % 16)
	{
		dim3 grid, block;

		if(0 == NV12frame.height() % 8)
		{
			grid = dim3(NV12frame.width() / 128, NV12frame.height() / 8);
			block = dim3(16, 8);
		}
		else
		{
			std::cout << NV12frame.height() << " % 8 != 0" << std::endl;
			return GPUFrame();
		}

		kernelNV12toRGBA<<< grid, block >>>(NV12frame.data(), NV12frame.pitch(), outputFrame.data(), outputFrame.pitch());
		cudaDeviceSynchronize(); cudaErr(cudaGetLastError());

		return outputFrame;
	}
	else
	{
		std::cout << "frame given didn't have 8*k width" << std::endl;
		return GPUFrame();
	}
}