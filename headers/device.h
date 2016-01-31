#ifndef DEVICE_H
#define DEVICE_H

/* 
 * Notes:
 *     'RGBA' refers to the typical format of
 *     red-green-blue-alpha pixel data.
 */

#include "GPUFrame.h"

// more explicit way of specifying data
typedef uint8_t byte;
typedef int8_t sbyte;
typedef uint64_t word; // optimal reads are 64 bits

// prototypes for interface functions (no need to open up kernel protoypes)

/* 
 * for every function meant to take in one frame and output another:
 * first prototype will allocate new space for the output frame,
 * second prototype will use space already allocated in 'RGBframe'
 */

// generic form of aforementioned prototypes:
// GPUFrame +fxnName+(GPUFrame& +format+input);
// int +fxnName+(GPUFrame& +format+input, GPUFrame& +newFormat+output)

// returns an object referring to the output RGBA image (format defined above) in device memory
GPUFrame NV12toRGBA(GPUFrame& NV12input); 
int NV12toRGBA(GPUFrame& NV12input, GPUFrame& RGBAoutput);

GPUFrame NV12toGrayscale(GPUFrame& NV12input); 
int NV12toGrayscale(GPUFrame& NV12input, GPUFrame& grayOutput);

GPUFrame RGBAtoGrayscale(GPUFrame& RGBAinput); 
int RGBAtoGrayscale(GPUFrame& RGBAinput, GPUFrame& grayOutput);

GPUFrame sobelFilter(GPUFrame& image);
int sobelFilter(GPUFrame& image, GPUFrame& edges);

GPUFrame matrixDifference(GPUFrame& positive, GPUFrame& negative);
int matrixDifference(GPUFrame& positive, GPUFrame& negative, GPUFrame& difference);

#endif
