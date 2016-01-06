#include <cuda_runtime.h>
#include "headers/device.h"

// development
#include <iostream>

#define YUV_TO_RGB_FLOAT

__device__ // function for GPU thread to use
inline void YUVtoRGBAmatrix(const byte& Y, const byte& U, const byte& V, byte& R, byte& G, byte& B, byte& A)
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

	// no alpha data in YUV
	A = 255;
}

// set 'clampCoords' to false when grid size
// perfectly corresponds to image size
// (each thread is 8 pixels wide)
template<bool clampCoords=false>
__global__
void kernelNV12toRGBA(const void* const input, const unsigned pitchInput,
					 void* const output, const unsigned pitchOutput,
					 const unsigned pixelsWidth=0, const unsigned pixelsHeight=0)
{
	// make sure we get the right data for clamping coords if necessary
	if(clampCoords)
		if(!(pixelsWidth && pixelsHeight))
			return;

	// dimensions of the grid
	// (# of blocks) * (threads per block)
	const unsigned gridHeight = gridDim.y * blockDim.y;
	// const unsigned gridWidth = gridDim.x * blockDim.x; // not in use

	// position within the grid
	// (threads per block) * (position of block in grid) + (position of thread in block)
	const unsigned gridXidx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned gridYidx = blockDim.y * blockIdx.y + threadIdx.y;

	// NV12 global reads for 8 pixels of data
	// address calculation from inner to outer access:
	// convert to byte array, position to proper row,
	// convert to row array, position to proper column
	word packed_Y_bytes, packed_UV_bytes;
	if(!clampCoords) // perfect grid size
	{
		packed_Y_bytes = reinterpret_cast<const word*>(static_cast<const byte*>(input) + gridYidx * pitchInput)[gridXidx];
		packed_UV_bytes = reinterpret_cast<const word*>(static_cast<const byte*>(input) + (gridHeight + gridYidx / 2) * pitchInput)[gridXidx];
	}
	else // wasted threads in some blocks
	{
		// keep waste threads from reading
	}

	// array representation of input data
	const byte* const Y  = reinterpret_cast<const byte*>(&packed_Y_bytes);
	const byte* const UV = reinterpret_cast<const byte*>(&packed_UV_bytes);

	// local destination for conversion
	word pixelPairs[4]; // pack into 4 2-pixel/8-byte pairs
	byte* const pair_0_bytes = reinterpret_cast<byte*>(pixelPairs + 0);
	byte* const pair_1_bytes = reinterpret_cast<byte*>(pixelPairs + 1);
	byte* const pair_2_bytes = reinterpret_cast<byte*>(pixelPairs + 2);
	byte* const pair_3_bytes = reinterpret_cast<byte*>(pixelPairs + 3);

	// this method of computation exposes ILP
	// convert all of the pixel data
	YUVtoRGBAmatrix(Y[0], UV[0], UV[1], pair_0_bytes[0], pair_0_bytes[1], pair_0_bytes[2], pair_0_bytes[3]);
	YUVtoRGBAmatrix(Y[1], UV[0], UV[1], pair_0_bytes[4], pair_0_bytes[5], pair_0_bytes[6], pair_0_bytes[7]);
	YUVtoRGBAmatrix(Y[2], UV[2], UV[3], pair_1_bytes[0], pair_1_bytes[1], pair_1_bytes[2], pair_1_bytes[3]);
	YUVtoRGBAmatrix(Y[3], UV[2], UV[3], pair_1_bytes[4], pair_1_bytes[5], pair_1_bytes[6], pair_1_bytes[7]);
	YUVtoRGBAmatrix(Y[4], UV[4], UV[5], pair_2_bytes[0], pair_2_bytes[1], pair_2_bytes[2], pair_2_bytes[3]);
	YUVtoRGBAmatrix(Y[5], UV[4], UV[5], pair_2_bytes[4], pair_2_bytes[5], pair_2_bytes[6], pair_2_bytes[7]);
	YUVtoRGBAmatrix(Y[6], UV[6], UV[7], pair_3_bytes[0], pair_3_bytes[1], pair_3_bytes[2], pair_3_bytes[3]);
	YUVtoRGBAmatrix(Y[7], UV[6], UV[7], pair_3_bytes[4], pair_3_bytes[5], pair_3_bytes[6], pair_3_bytes[7]);

	// strided global write of the RGBA data for 8 pixels,
	// taking the hit on efficiency
	word* const row = reinterpret_cast<word*>(static_cast<byte*>(output) + gridYidx * pitchInput);
	const unsigned firstColumn = 4 * gridXidx;
	row[firstColumn    ] = pixelPairs[0];
	row[firstColumn + 1] = pixelPairs[1];
	row[firstColumn + 2] = pixelPairs[2];
	row[firstColumn + 3] = pixelPairs[3];
}

// (maybe) when this works, modify it to push to a ConcurrentQueue<GPUFrame>
// allocate new space before converting
GPUFrame NV12toRGBA(GPUFrame& NV12input)
{
	// reference for the new frame
	GPUFrame allocatedFrame;

	// make an object for the output image
	unsigned allocationRows = 4 * NV12input.height();
	unsigned allocationCols = NV12input.width();

	// make the actual memory allocation
	allocatedFrame = GPUFrame(NV12input.width(), NV12input.height(), allocationCols, allocationRows, NV12input.timestamp());

	if(0 == NV12toRGBA(NV12input, allocatedFrame))
	{
		// original success indicator
		return allocatedFrame;
	}
	else
	{
		// original failure indicator
		return GPUFrame();
	}
}

// run conversion kernel with pre-allocated output memory
// return 0 on success, anything else on failure
int NV12toRGBA(GPUFrame& NV12input, GPUFrame& RGBAoutput)
{
	// make sure the width divides nicely
	if(0 == NV12input.width() % 16)
	{
		dim3 grid, block;

		// make sure the height divides nicely
		if(0 == NV12input.height() % 8)
		{
			// make dimension objects for kernel launch
			grid = dim3(NV12input.width() / 128, NV12input.height() / 8);
			block = dim3(16, 8);
		}
		else
		{
			std::cout << NV12input.height() << " % 8 != 0" << std::endl;
			return 1; // failure
		}

		// GPU computation
		kernelNV12toRGBA<<< grid, block >>>(NV12input.data(), NV12input.pitch(), RGBAoutput.data(), RGBAoutput.pitch());
		cudaDeviceSynchronize(); cudaErr(cudaGetLastError());
	}
	else
	{
		std::cout << NV12input.width() << " % 16 != 0" << std::endl;
		return 1; // failure
	}

	return 0; // success
}
