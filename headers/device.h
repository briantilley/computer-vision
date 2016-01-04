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

// returns an object referring to the output RGBA image in device memory
GPUFrame NV12toRGB(GPUFrame& NV12input, bool makeAlpha=false);

#endif