#include <cuda_runtime.h>
#include "headers/device.h"

// development
#include <iostream>
#include <cuda_profiler_api.h>

// #define YUV_TO_RGB_FLOAT

template<int pAidx = -1> // pointer to A in case it is to be written
__device__ // function for GPU thread to use
inline void YUVtoRGBmatrix(const byte& Y, const byte& U, const byte& V, byte& R, byte& G, byte& B, byte* pA = nullptr)
{
	// convert and clamp values between 0 and 255 inclusive
	#ifdef YUV_TO_RGB_FLOAT
		R = min(max(static_cast<int>(Y + 1.28033f * V + .5f), 0), 255);
		G = min(max(static_cast<int>(Y - 0.21482f * U - 0.38059f * V + .5f), 0), 255);
		B = min(max(static_cast<int>(Y + 2.12798f * U + .5f), 0), 255);
	#else // integer math makes kernel slightly more than 9us/9.7% faster overall
		R = min(max(Y + ((328 * V) >> 8), 0), 255);
		G = min(max(Y - ((55 * U + 97 * V) >> 8), 0), 255);
		B = min(max(Y + ((545 * U) >> 8), 0), 255);
	#endif

	// always set opacity to full if specified,
	// this is left out of compilation otherwise
	if(-1 != pAidx)
		pA[pAidx] = 255;
}

// use this when grid size perfectly matches image size (each thread is 4 pixels)
template<bool writeAlpha=false>
__global__
void kernelNV12toRGB(const void* const input, const unsigned pitchInput,
					  void* const output, const unsigned pitchOutput)
{
	// dimensions of the grid
	// (# of blocks) * (threads per block)
	const unsigned gridHeight = gridDim.y * blockDim.y;
	// const unsigned gridWidth = gridDim.x * blockDim.x; // not in use

	// position within the grid
	// (threads per block) * (position of block in grid) + (position of thread in block)
	const unsigned gridXidx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned gridYidx = blockDim.y * blockIdx.y + threadIdx.y;

	// NV12 global reads for 8 pixels
	// address calculation from inner to outer access:
	// convert to byte array, position to proper row,
	// convert to row array, position to proper column
	const word packedYbytes = reinterpret_cast<const word*>(static_cast<const byte*>(input) + gridYidx * pitchInput)[gridXidx];
	const word packedUVbytes = reinterpret_cast<const word*>(static_cast<const byte*>(input) + (gridHeight + gridYidx / 2) * pitchInput)[gridXidx];

	// local destination for conversion
	word packedRbytes = 0, packedGbytes = 0, packedBbytes = 0, packedAbytes = 0;
	
	// make byte arrays out of the packed words
	const byte* const Y  = reinterpret_cast<const byte*>(&packedYbytes);
	const byte* const UV = reinterpret_cast<const byte*>(&packedUVbytes);
	byte* const R = reinterpret_cast<byte*>(&packedRbytes);
	byte* const G = reinterpret_cast<byte*>(&packedGbytes);
	byte* const B = reinterpret_cast<byte*>(&packedBbytes);

	// this method of sequence exposes ILP
	// convert all of the pixel data
	if(writeAlpha)
	{
		byte* const pA = reinterpret_cast<byte*>(&packedAbytes);
		YUVtoRGBmatrix<0>(Y[0], UV[0], UV[1], R[0], G[0], B[0], pA);
		YUVtoRGBmatrix<1>(Y[1], UV[0], UV[1], R[1], G[1], B[1], pA);
		YUVtoRGBmatrix<2>(Y[2], UV[2], UV[3], R[2], G[2], B[2], pA);
		YUVtoRGBmatrix<3>(Y[3], UV[2], UV[3], R[3], G[3], B[3], pA);
		YUVtoRGBmatrix<4>(Y[4], UV[4], UV[5], R[4], G[4], B[4], pA);
		YUVtoRGBmatrix<5>(Y[5], UV[4], UV[5], R[5], G[5], B[5], pA);
		YUVtoRGBmatrix<6>(Y[6], UV[6], UV[7], R[6], G[6], B[6], pA);
		YUVtoRGBmatrix<7>(Y[7], UV[6], UV[7], R[7], G[7], B[7], pA);
	}
	else
	{
		YUVtoRGBmatrix(Y[0], UV[0], UV[1], R[0], G[0], B[0]);
		YUVtoRGBmatrix(Y[1], UV[0], UV[1], R[1], G[1], B[1]);
		YUVtoRGBmatrix(Y[2], UV[2], UV[3], R[2], G[2], B[2]);
		YUVtoRGBmatrix(Y[3], UV[2], UV[3], R[3], G[3], B[3]);
		YUVtoRGBmatrix(Y[4], UV[4], UV[5], R[4], G[4], B[4]);
		YUVtoRGBmatrix(Y[5], UV[4], UV[5], R[5], G[5], B[5]);
		YUVtoRGBmatrix(Y[6], UV[6], UV[7], R[6], G[6], B[6]);
		YUVtoRGBmatrix(Y[7], UV[6], UV[7], R[7], G[7], B[7]);
	}

	// coalesced global write of the RGB(A) data for 8 pixels
	// address calculation from inner to outer access:
	// convert to byte array, position to proper row,
	// convert to row array, position to proper column
	reinterpret_cast<word*>(static_cast<byte*>(output) + gridYidx * pitchInput)[gridXidx] = packedRbytes;
	reinterpret_cast<word*>(static_cast<byte*>(output) + (gridHeight + gridYidx) * pitchInput)[gridXidx] = packedGbytes;
	reinterpret_cast<word*>(static_cast<byte*>(output) + (2 * gridHeight + gridYidx) * pitchInput)[gridXidx] = packedBbytes;
	if(writeAlpha)
		reinterpret_cast<word*>(static_cast<byte*>(output) + (3 * gridHeight + gridYidx) * pitchInput)[gridXidx] = packedAbytes;
}

// use this when grid has threads outside the image
template<bool writeAlpha=false>
__global__
void paddedKernelNV12toRGB(const word* const input, const unsigned pitchInput,
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

		if(makeAlpha)
			kernelNV12toRGB<true><<< grid, block >>>(NV12frame.data(), NV12frame.pitch(), outputFrame.data(), outputFrame.pitch());
		else
			kernelNV12toRGB<false><<< grid, block >>>(NV12frame.data(), NV12frame.pitch(), outputFrame.data(), outputFrame.pitch());

		cudaDeviceSynchronize(); cudaErr(cudaGetLastError());

		return outputFrame;
	}
	else
	{
		std::cout << "frame given didn't have 8*k width" << std::endl;
		return GPUFrame();
	}
}