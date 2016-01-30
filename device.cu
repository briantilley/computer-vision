#include <cuda_runtime.h>
#include "headers/device.h"

// development
#include <iostream>

// #define YUV_TO_RGB_FLOAT

// convert to RGB using ITU 601 standard
__device__ // function for GPU thread to use
inline void YUVtoRGBA(const byte& Y, const byte& U, const byte& V, byte& R, byte& G, byte& B, byte& A)
{
	#ifdef YUV_TO_RGB_FLOAT
		float fY, fU, fV;

		fY = Y * 298.082f / 256.f;
		fU = U / 256.f;
		fV = V / 256.f;

		R = min(max(static_cast<int>(fY + 408.593f * fV - 222.921), 0), 255);
		G = min(max(static_cast<int>(fY - 100.291f * fU - 208.12f * fV + 135.576), 0), 255);
		B = min(max(static_cast<int>(fY + 516.412 * fU - 276.836), 0), 255);
	#else
		int tY = 298 * Y;

		R = min(max((tY + 409 * V - 57068) >> 8, 0), 255);
		G = min(max((tY - 100 * U - 208 * V + 34707 >> 8), 0), 255);
		B = min(max((tY + 516 * U - 70870) >> 8, 0), 255);
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
	{
		if(!(pixelsWidth && pixelsHeight))
			return;
	}

	// // number of rows of threads in use
	// const unsigned gridWidth = gridDim.x * blockDim.x;
	unsigned gridHeight;
	if(clampCoords)
		gridHeight = pixelsHeight;
	else
		gridHeight = gridDim.y * blockDim.y;

	// position within the grid
	// (threads per block) * (position of block in grid) + (position of thread in block)
	const unsigned gridXidx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned gridYidx = blockDim.y * blockIdx.y + threadIdx.y;

	if(clampCoords)
	{
		if(gridXidx * 8 >= pixelsWidth || gridYidx >= pixelsHeight)
			return;
	}

	// NV12 global reads for 8 pixels of data
	// address calculation from inner to outer access:
	// convert to byte array, position to proper row,
	// convert to row array, position to proper column
	word packed_Y_bytes, packed_UV_bytes;
	packed_Y_bytes = reinterpret_cast<const word*>(static_cast<const byte*>(input) + gridYidx * pitchInput)[gridXidx];
	packed_UV_bytes = reinterpret_cast<const word*>(static_cast<const byte*>(input) + (gridHeight + gridYidx / 2) * pitchInput)[gridXidx];

	// array representation of input data
	const byte* const Y  = reinterpret_cast<const byte*>(&packed_Y_bytes);
	const byte* const UV = reinterpret_cast<const byte*>(&packed_UV_bytes);

	// local destination for conversion
	word pixelPairs[4]; // pack into 4 2-pixel/8-byte pairs
	byte* const pair_0_bytes = reinterpret_cast<byte*>(pixelPairs    );
	byte* const pair_1_bytes = reinterpret_cast<byte*>(pixelPairs + 1);
	byte* const pair_2_bytes = reinterpret_cast<byte*>(pixelPairs + 2);
	byte* const pair_3_bytes = reinterpret_cast<byte*>(pixelPairs + 3);

	// this method of computation exposes ILP
	// convert all of the pixel data
	YUVtoRGBA(Y[0], UV[0], UV[1], pair_0_bytes[0], pair_0_bytes[1], pair_0_bytes[2], pair_0_bytes[3]);
	YUVtoRGBA(Y[1], UV[0], UV[1], pair_0_bytes[4], pair_0_bytes[5], pair_0_bytes[6], pair_0_bytes[7]);
	YUVtoRGBA(Y[2], UV[2], UV[3], pair_1_bytes[0], pair_1_bytes[1], pair_1_bytes[2], pair_1_bytes[3]);
	YUVtoRGBA(Y[3], UV[2], UV[3], pair_1_bytes[4], pair_1_bytes[5], pair_1_bytes[6], pair_1_bytes[7]);
	YUVtoRGBA(Y[4], UV[4], UV[5], pair_2_bytes[0], pair_2_bytes[1], pair_2_bytes[2], pair_2_bytes[3]);
	YUVtoRGBA(Y[5], UV[4], UV[5], pair_2_bytes[4], pair_2_bytes[5], pair_2_bytes[6], pair_2_bytes[7]);
	YUVtoRGBA(Y[6], UV[6], UV[7], pair_3_bytes[0], pair_3_bytes[1], pair_3_bytes[2], pair_3_bytes[3]);
	YUVtoRGBA(Y[7], UV[6], UV[7], pair_3_bytes[4], pair_3_bytes[5], pair_3_bytes[6], pair_3_bytes[7]);

	// strided global write of the RGBA data for 8 pixels,
	// taking the hit on efficiency
	word* const row = reinterpret_cast<word*>(static_cast<byte*>(output) + gridYidx * pitchOutput);
	const unsigned firstColumn = 4 * gridXidx; 

	row[firstColumn    ] = pixelPairs[0];
	row[firstColumn + 1] = pixelPairs[1];
	row[firstColumn + 2] = pixelPairs[2];
	row[firstColumn + 3] = pixelPairs[3];
}

// wil be 8 pixels per thread, load 2 at a time with grid-strided reads
// arrange read and write such that the one memory location can be used
// kernel must be given normalized convolution matrices of odd width and height
__global__
void kernelMatrixConvolution(const void* const input, const unsigned pitchInput,
							 void* const output, const unsigned pitchOutput,
							 const unsigned pixelsWidth, const unsigned pixelsHeight,
							 const float* const filterMatrix,
							 const unsigned filterMatrixWidth, const unsigned filterMatrixHeight)
{
	// // dimensions of the grid
	// const unsigned gridWidth = gridDim.x * blockDim.x;
	// const unsigned gridHeight = gridDim.y * blockDim.y;

	// indices of each thread
	const unsigned gridXidx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned gridYidx = blockIdx.y * blockDim.y + threadIdx.y;

	// kill threads that are out of bounds
	if(gridXidx >= pixelsWidth || gridYidx >= pixelsHeight)
		return;

	int newVal = 0; // new RGBA values for the two pixels
	float readVal = 0; // two pixels read from input

	const int firstRow = gridYidx - filterMatrixHeight / 2;
	const int firstColumn = gridXidx - filterMatrixWidth / 2;

	for(int colorElement = 0; colorElement < 3; ++colorElement)
	{
		newVal = 0;

		for(int i = 0; i < filterMatrixHeight; ++i)
		{
			for(int j = 0; j < filterMatrixWidth; ++j)
			{
				readVal = static_cast<const byte*>(input)[min(max(firstRow + i, 0), pixelsHeight - 1) * pitchInput + min(max(firstColumn + j, 0), pixelsWidth - 1) * 4 + colorElement];
				newVal += readVal * filterMatrix[i * filterMatrixWidth + j] + 0.5f;
			}
		}

		static_cast<byte*>(output)[gridYidx * pitchOutput + gridXidx * 4 + colorElement] = min(min(newVal, 0), 255);
	}
}

// sum of magnitudes of A and B treated as legs of a right triangle
// should be 2 pixels per thread
__global__
void kernelVectorSum(const void* const inputA, const unsigned pitchInputA,
					 const void* const inputB, const unsigned pitchInputB,
					 void* const output, const unsigned pitchOutput,
					 const unsigned pixelsWidth, const unsigned pixelsHeight)
{
	const unsigned gridWidth = gridDim.x * blockDim.x;

	// indices of each thread
	const unsigned gridXidx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned gridYidx = blockIdx.y * blockDim.y + threadIdx.y;

	// kill threads that are out of bounds
	if(/*gridXidx * 2 >= pixelsWidth || */gridYidx >= pixelsHeight)
		return;

	// destinations for packed data
	word inputApixels = 0, inputBpixels = 0, outputCpixels = 0;

	// read
	inputApixels = reinterpret_cast<const word*>(static_cast<const byte*>(inputA) + gridYidx * pitchInputA)[gridXidx];
	inputBpixels = reinterpret_cast<const word*>(static_cast<const byte*>(inputB) + gridYidx * pitchInputB)[gridXidx];

	if(gridXidx >= gridWidth / 2)
	{
		// inputApixels = 0xffffffffffffffff;
		// inputBpixels = 0xffffffffffffffff;
	}

	// store the vector magnitude of each component
	reinterpret_cast<byte*>(&outputCpixels)[0] = sqrt(static_cast<float>(reinterpret_cast<const byte*>(&inputApixels)[0]) * reinterpret_cast<const byte*>(&inputApixels)[0] + static_cast<float>(reinterpret_cast<const byte*>(&inputBpixels)[0]) * reinterpret_cast<const byte*>(&inputBpixels)[0]);
	reinterpret_cast<byte*>(&outputCpixels)[1] = sqrt(static_cast<float>(reinterpret_cast<const byte*>(&inputApixels)[1]) * reinterpret_cast<const byte*>(&inputApixels)[1] + static_cast<float>(reinterpret_cast<const byte*>(&inputBpixels)[1]) * reinterpret_cast<const byte*>(&inputBpixels)[1]);
	reinterpret_cast<byte*>(&outputCpixels)[2] = sqrt(static_cast<float>(reinterpret_cast<const byte*>(&inputApixels)[2]) * reinterpret_cast<const byte*>(&inputApixels)[2] + static_cast<float>(reinterpret_cast<const byte*>(&inputBpixels)[2]) * reinterpret_cast<const byte*>(&inputBpixels)[2]);

	reinterpret_cast<byte*>(&outputCpixels)[4] = sqrt(static_cast<float>(reinterpret_cast<const byte*>(&inputApixels)[4]) * reinterpret_cast<const byte*>(&inputApixels)[4] + static_cast<float>(reinterpret_cast<const byte*>(&inputBpixels)[4]) * reinterpret_cast<const byte*>(&inputBpixels)[4]);
	reinterpret_cast<byte*>(&outputCpixels)[5] = sqrt(static_cast<float>(reinterpret_cast<const byte*>(&inputApixels)[5]) * reinterpret_cast<const byte*>(&inputApixels)[5] + static_cast<float>(reinterpret_cast<const byte*>(&inputBpixels)[5]) * reinterpret_cast<const byte*>(&inputBpixels)[5]);
	reinterpret_cast<byte*>(&outputCpixels)[6] = sqrt(static_cast<float>(reinterpret_cast<const byte*>(&inputApixels)[6]) * reinterpret_cast<const byte*>(&inputApixels)[6] + static_cast<float>(reinterpret_cast<const byte*>(&inputBpixels)[6]) * reinterpret_cast<const byte*>(&inputBpixels)[6]);

	// write
	reinterpret_cast<word*>(static_cast<byte*>(output) + gridYidx * pitchOutput)[gridXidx] = outputCpixels;
}

// (maybe) when this works, modify it to push to a ConcurrentQueue<GPUFrame>
// allocate new space before converting
GPUFrame NV12toRGBA(GPUFrame& NV12input)
{
	// reference for the new frame
	GPUFrame allocatedFrame;

	// make an object for the output image
	unsigned allocationRows = NV12input.height();
	unsigned allocationCols = 4 * NV12input.width();

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
// TODO: switch statement for common sizes and template call for ones needing padding
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 8
int NV12toRGBA(GPUFrame& NV12input, GPUFrame& RGBAoutput)
{
	// make sure the width and height divide nicely
	bool matchedWidth = !(NV12input.width() % (8 * BLOCK_WIDTH));
	bool matchedHeight = !(NV12input.height() % BLOCK_HEIGHT);

	if(matchedWidth && matchedHeight)
	{
		// dimensions for kernel launch
		dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
		dim3 grid(NV12input.width() / (8 * block.x), NV12input.height() / block.y);

		kernelNV12toRGBA<false><<< grid, block >>>(NV12input.data(), NV12input.pitch(),
												   RGBAoutput.data(), RGBAoutput.pitch());

		// sync and check for errors
		cudaDeviceSynchronize(); cudaErr(cudaGetLastError());
	}
	else
	{
		// dimensions for kernel launch
		dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
		dim3 grid(NV12input.width() / (8 * block.x), NV12input.height() / block.y);

		// add in a block of width and/or height to reach all pixels
		if(!matchedWidth)
			grid.x++;

		if(!matchedHeight)
			grid.y++;

		kernelNV12toRGBA<true><<< grid, block >>>(NV12input.data(), NV12input.pitch(),
												  RGBAoutput.data(), RGBAoutput.pitch(),
												  RGBAoutput.width(), RGBAoutput.height());

		// sync and check for errors
		cudaDeviceSynchronize(); cudaErr(cudaGetLastError());
	}

	return 0; // success
}

// allocate for and run the sobel filter
GPUFrame sobelFilter(GPUFrame& image)
{
	// reference for the new frame
	GPUFrame allocatedFrame;

	// make an object for the output image
	unsigned allocationRows = image.height();
	unsigned allocationCols = 4 * image.width();

	// make the actual memory allocation
	allocatedFrame = GPUFrame(image.width(), image.height(), allocationCols, allocationRows, image.timestamp());

	if(0 == sobelFilter(image, allocatedFrame))
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

// launch sobel filter kernel
int sobelFilter(GPUFrame& image, GPUFrame& edges)
{
	// keep static device pointer to normalized sobel
	// convolution filter and generate if first call
	static float hostSobelXFilter[] = {-1.f/8, 0.f, 1.f/8, -2.f/8, 0.f, 2.f/8, -1.f/8, 0.f, 1.f/8};
	static float* sobelXFilter = nullptr;
	static float hostSobelYFilter[] = {-1.f/8, -2.f/8, -1.f/8, 0, 0, 0, 1.f/8, 2.f/8, 1.f/8};
	static float* sobelYFilter = nullptr;

	if(nullptr == sobelXFilter)
	{
		cudaErr(cudaMalloc(&sobelXFilter, 9 * sizeof(float)));
		cudaErr(cudaMemcpy(sobelXFilter, hostSobelXFilter, 9 * sizeof(float), cudaMemcpyHostToDevice));

		cudaErr(cudaMalloc(&sobelYFilter, 9 * sizeof(float)));
		cudaErr(cudaMemcpy(sobelYFilter, hostSobelYFilter, 9 * sizeof(float), cudaMemcpyHostToDevice));
	}

	static GPUFrame sobelX(image.width(), image.height(), 4 * image.width(), image.height(), 0);
	static GPUFrame sobelY(image.width(), image.height(), 4 * image.width(), image.height(), 0);

	// figure out dimensions
	dim3 grid, block(BLOCK_WIDTH, BLOCK_HEIGHT);
	grid.x = image.width() / (1 * block.x);
	grid.y = image.height() / block.y;

	if(image.width() % (1 * block.x))
		grid.x++;

	if(image.height() % block.y)
		grid.y++;

	// launch convolution kernel with sobel matrix
	kernelMatrixConvolution<<< grid, block >>>(image.data(), image.pitch(),
											   sobelX.data(), sobelX.pitch(),
											   image.width(), image.height(),
											   sobelXFilter,
											   3, 3);

	kernelMatrixConvolution<<< grid, block >>>(image.data(), image.pitch(),
											   sobelY.data(), sobelY.pitch(),
											   image.width(), image.height(),
											   sobelYFilter,
											   3, 3);

	// vector sum of both sobel images
	kernelVectorSum<<< grid, block >>>(sobelX.data(), sobelX.pitch(),
									   sobelY.data(), sobelY.pitch(),
									   edges.data(), edges.pitch(),
									   image.width(), image.height());

	// sync and check for errors
	cudaDeviceSynchronize(); cudaErr(cudaGetLastError());

	// success
	return 0;
}