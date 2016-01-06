#include <cuda_runtime.h>
#include "headers/device.h"

// development
#include <iostream>

// #define YUV_TO_RGB_FLOAT

template<int pAidx=-1> // pointer to A in case it is to be written
__device__ // function for GPU thread to use
inline void YUVtoRGBmatrix(const byte& Y, const byte& U, const byte& V, byte& R, byte& G, byte& B, byte* pA = nullptr)
{
	// make sure we get the necessary address if writing alpha data
	if(-1 != pAidx)
		if(nullptr == pA)
			return;

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

// set 'clampCoords' to false when grid size
// perfectly corresponds to image size
// (each thread is 8 pixels wide)
template<bool clampCoords=false, bool writeAlpha=false>
__global__
void kernelNV12toRGB(const void* const input, const unsigned pitchInput,
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

	// NV12 global reads for 8 pixels
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
		// make waste threads read from the nearest edge
	}

	// local destination for conversion
	word packed_R_bytes = 0, packed_G_bytes = 0, packed_B_bytes = 0, packed_A_bytes = 0;
	
	// make byte arrays out of the packed words
	const byte* const Y  = reinterpret_cast<const byte*>(&packed_Y_bytes);
	const byte* const UV = reinterpret_cast<const byte*>(&packed_UV_bytes);
	byte* const R = reinterpret_cast<byte*>(&packed_R_bytes);
	byte* const G = reinterpret_cast<byte*>(&packed_G_bytes);
	byte* const B = reinterpret_cast<byte*>(&packed_B_bytes);

	// this method of sequence exposes ILP
	// convert all of the pixel data
	if(writeAlpha)
	{
		byte* const pA = reinterpret_cast<byte*>(&packed_A_bytes);
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
	reinterpret_cast<word*>(static_cast<byte*>(output) + gridYidx * pitchInput)[gridXidx] = packed_R_bytes;
	reinterpret_cast<word*>(static_cast<byte*>(output) + (gridHeight + gridYidx) * pitchInput)[gridXidx] = packed_G_bytes;
	reinterpret_cast<word*>(static_cast<byte*>(output) + (2 * gridHeight + gridYidx) * pitchInput)[gridXidx] = packed_B_bytes;
	if(writeAlpha)
		reinterpret_cast<word*>(static_cast<byte*>(output) + (3 * gridHeight + gridYidx) * pitchInput)[gridXidx] = packed_A_bytes;
}

// convert struct-of-arrays type RGB to array-of-structs type RGBA,
// set 'clampCoords' to false when grid size
// perfectly corresponds to image size
// (each thread is 8 pixels wide)
template<bool clampCoords=false, bool readAlpha=false>
__global__
void kernelSurfaceRGBtoRGBA(const void* const input, const unsigned pitchInput,
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

	// global reads for 8 pixels from struct-of-arrays type RGB(A) frame
	// reading 8 bytes at a time makes for optimal performance,
	// default opacity is 255
	word packed_R_bytes = 0, packed_G_bytes = 0, packed_B_bytes = 0, packed_A_bytes = 0xffffffffffffffff;
	if(!clampCoords) // perfect grid size
	{
		packed_R_bytes = reinterpret_cast<const word*>(static_cast<const byte*>(input) + gridYidx * pitchInput)[gridXidx];
		packed_G_bytes = reinterpret_cast<const word*>(static_cast<const byte*>(input) + (gridHeight + gridYidx) * pitchInput)[gridXidx];
		packed_B_bytes = reinterpret_cast<const word*>(static_cast<const byte*>(input) + (2 * gridHeight + gridYidx) * pitchInput)[gridXidx];
		if(readAlpha) // alpha channel isn't always present
			packed_A_bytes = reinterpret_cast<const word*>(static_cast<const byte*>(input) + (3 * gridHeight + gridYidx) * pitchInput)[gridXidx];
	}
	else // wasted threads in some blocks
	{
		// make waste threads read from the nearest edge
	}

	word pixelPairs[4] = {0, 0, 0, 0}; // pack into 4 2-pixel/8-byte pairs
	byte* pair_0_bytes = reinterpret_cast<byte*>(pixelPairs + 0);
	byte* pair_1_bytes = reinterpret_cast<byte*>(pixelPairs + 1);
	byte* pair_2_bytes = reinterpret_cast<byte*>(pixelPairs + 2);
	byte* pair_3_bytes = reinterpret_cast<byte*>(pixelPairs + 3);

	byte* R_bytes = reinterpret_cast<byte*>(&packed_R_bytes);
	byte* G_bytes = reinterpret_cast<byte*>(&packed_G_bytes);
	byte* B_bytes = reinterpret_cast<byte*>(&packed_B_bytes);
	byte* A_bytes = reinterpret_cast<byte*>(&packed_A_bytes);

	// optimize ILP exposure

	// red values
	pair_0_bytes[0] = R_bytes[0]; // pixel 1
	pair_1_bytes[0] = R_bytes[2]; // pixel 3
	pair_2_bytes[0] = R_bytes[4]; // pixel 5
	pair_3_bytes[0] = R_bytes[6]; // pixel 7
	pair_0_bytes[4] = R_bytes[1]; // pixel 2
	pair_1_bytes[4] = R_bytes[3]; // pixel 4
	pair_2_bytes[4] = R_bytes[5]; // pixel 6
	pair_3_bytes[4] = R_bytes[7]; // pixel 8
	// green values
	pair_0_bytes[1] = G_bytes[0]; // pixel 1
	pair_1_bytes[1] = G_bytes[2]; // pixel 3
	pair_2_bytes[1] = G_bytes[4]; // pixel 5
	pair_3_bytes[1] = G_bytes[6]; // pixel 7
	pair_0_bytes[5] = G_bytes[1]; // pixel 2
	pair_1_bytes[5] = G_bytes[3]; // pixel 4
	pair_2_bytes[5] = G_bytes[5]; // pixel 6
	pair_3_bytes[5] = G_bytes[7]; // pixel 8
	// blue values
	pair_0_bytes[2] = B_bytes[0]; // pixel 1
	pair_1_bytes[2] = B_bytes[2]; // pixel 3
	pair_2_bytes[2] = B_bytes[4]; // pixel 5
	pair_3_bytes[2] = B_bytes[6]; // pixel 7
	pair_0_bytes[6] = B_bytes[1]; // pixel 2
	pair_1_bytes[6] = B_bytes[3]; // pixel 4
	pair_2_bytes[6] = B_bytes[5]; // pixel 6
	pair_3_bytes[6] = B_bytes[7]; // pixel 8
	// red values
	pair_0_bytes[3] = A_bytes[0]; // pixel 1
	pair_1_bytes[3] = A_bytes[2]; // pixel 3
	pair_2_bytes[3] = A_bytes[4]; // pixel 5
	pair_3_bytes[3] = A_bytes[6]; // pixel 7
	pair_0_bytes[7] = A_bytes[1]; // pixel 2
	pair_1_bytes[7] = A_bytes[3]; // pixel 4
	pair_2_bytes[7] = A_bytes[5]; // pixel 6
	pair_3_bytes[7] = A_bytes[7]; // pixel 8

	// only things left to do is write (conversion just moves data)
	word* outputRow = reinterpret_cast<word*>(static_cast<byte*>(output) + gridYidx * pitchOutput);
	const unsigned firstPosition = 4 * gridXidx;
	if(!clampCoords) // perfect grid size
	{ // cudaErrorLaunchFailure from these writes
		outputRow[firstPosition] = pixelPairs[0];
		outputRow[firstPosition + 1] = pixelPairs[1];
		outputRow[firstPosition + 2] = pixelPairs[2];
		outputRow[firstPosition + 3] = pixelPairs[3];
	}
	else // wasted threads in some blocks
	{
		// keep waste threads from writing
	}
}

// (maybe) when this works, modify it to push to a ConcurrentQueue<GPUFrame>
// allocate new space before converting
GPUFrame NV12toRGB(GPUFrame& NV12input, const bool writeAlpha)
{
	GPUFrame allocatedFrame;

	// make an object for the output image
	unsigned allocationRows = (writeAlpha ? 4 : 3) * NV12input.height();
	unsigned allocationCols = NV12input.width();

	// make the actual memory allocation
	allocatedFrame = GPUFrame(NV12input.width(), NV12input.height(), allocationCols, allocationRows, NV12input.timestamp());

	if(0 == NV12toRGB(NV12input, allocatedFrame, writeAlpha))
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
int NV12toRGB(GPUFrame& NV12input, GPUFrame& RGBoutput, const bool writeAlpha)
{
	if(0 == NV12input.width() % 16)
	{
		dim3 grid, block;

		if(0 == NV12input.height() % 8)
		{
			grid = dim3(NV12input.width() / 128, NV12input.height() / 8);
			block = dim3(16, 8);
		}
		else
		{
			std::cout << NV12input.height() << " % 8 != 0" << std::endl;
			return 1;
		}

		if(writeAlpha)
			kernelNV12toRGB<false, true><<< grid, block >>>(NV12input.data(), NV12input.pitch(), RGBoutput.data(), RGBoutput.pitch());
		else
			kernelNV12toRGB<<< grid, block >>>(NV12input.data(), NV12input.pitch(), RGBoutput.data(), RGBoutput.pitch());

		cudaDeviceSynchronize(); cudaErr(cudaGetLastError());
	}
	else
	{
		std::cout << NV12input.width() << " % 16 != 0" << std::endl;
		return 1;
	}

	return 0;
}

// (maybe) when this works, modify it to push to a ConcurrentQueue<GPUFrame>
// allocate new space before converting
GPUFrame RGBtoRGBA(GPUFrame& RGBinput, const bool readAlpha)
{
	GPUFrame allocatedFrame;

	// make an object for the output image
	unsigned allocationRows = RGBinput.height();
	unsigned allocationCols = 4 * RGBinput.width();

	// make the actual memory allocation
	allocatedFrame = GPUFrame(RGBinput.width(), RGBinput.height(), allocationCols, allocationRows, RGBinput.timestamp());

	if(0 == RGBtoRGBA(RGBinput, allocatedFrame, readAlpha))
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
int RGBtoRGBA(GPUFrame& RGBinput, GPUFrame& RGBAoutput, const bool readAlpha)
{
	if(0 == RGBinput.width() % 16)
	{
		dim3 grid, block;

		if(0 == RGBinput.height() % 8)
		{
			grid = dim3(RGBinput.width() / 128, RGBinput.height() / 8);
			block = dim3(16, 8);
		}
		else
		{
			std::cout << RGBinput.height() << " % 8 != 0" << std::endl;
			return 1;
		}

		if(readAlpha)
			kernelSurfaceRGBtoRGBA<false, true><<< grid, block >>>(RGBinput.data(), RGBinput.pitch(), RGBAoutput.data(), RGBAoutput.pitch());
		else
			kernelSurfaceRGBtoRGBA<<< grid, block >>>(RGBinput.data(), RGBinput.pitch(), RGBAoutput.data(), RGBAoutput.pitch());

		cudaDeviceSynchronize(); cudaErr(cudaGetLastError());
	}
	else
	{
		std::cout << RGBinput.width() << " % 16 != 0" << std::endl;
		return 1;
	}

	return 0;
}
