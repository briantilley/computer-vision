#ifndef DEVICE_H
#define DEVICE_H

/* RGB and RGBA formats are modified for optimal conversion from NV12
 *
 * Description:
 * one surface per color parameter each as wide (in bytes) as the image width
 * and same height as the image (in rows), total of 3 or 4 times for RGB or RGBA,
 * respectively, as many rows as image height, same number of columns as image
 * width; first surface is red, second is green, third is blue, optional fourth
 * is alpha
 *
 * Graphical representation for a 5x4 pixel image:
 * R R R R R red
 * R R R R R
 * R R R R R
 * R R R R R
 * G G G G G green
 * G G G G G
 * G G G G G
 * G G G G G
 * B B B B B blue
 * B B B B B
 * B B B B B
 * B B B B B
 * A A A A A optional alpha surface
 * A A A A A
 * A A A A A
 * A A A A A
 */

#include "GPUFrame.h"

// more explicit way of specifying data
typedef uint8_t byte;
typedef uint64_t word; // optimal reads are 64 bits

// prototypes for interface functions (no need to open up kernel protoypes)

/* 
 * for every function meant to take in one frame and output another:
 * first prototype will allocate new space for the output frame,
 * second prototype will use space already allocated in 'RGBframe'
 */

// generic form of aforementioned prototypes:
// GPUFrame +fxnName+(GPUFrame& +format+input, const bool +read/write+Alpha=false);
// int +fxnName+(GPUFrame& +format+input, GPUFrame& +newFormat+output, const bool +read/write+Alpha=false)

// returns an object referring to the output RGB(A) image (format defined above) in device memory
GPUFrame NV12toRGB(GPUFrame& NV12input, const bool writeAlpha=false); 
int NV12toRGB(GPUFrame& NV12input, GPUFrame& RGBoutput, const bool writeAlpha=false);

// returns an object referring to the output RGBA (standard format) image in device memory
GPUFrame RGBtoRGBA(GPUFrame& RGBinput, const bool readAlpha=false);
int RGBtoRGBA(GPUFrame& RGBinput, GPUFrame& RGBAoutput, const bool readAlpha=false);

#endif