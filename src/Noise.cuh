#ifndef NOISE_CUH
#define NOISE_CUH

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>

#include "FileWriter.cuh"
#include "Map.cuh"
#include "RandomHelper.cuh"

#define MAXBLOCKS 8
#define BLOCKSIZE 256

float** noise(int width, int height);

__global__ void seedRandomizer(curandState** cs, int xDim, int yDim);

__global__ void noiseKernel(float** mapFragment, int delta, int width, int height, curandState** cs);

#endif
