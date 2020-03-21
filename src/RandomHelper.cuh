#ifndef RANDOMHELPER_CUH
#define RANDOMHELPER_CUH

#include <curand.h>
#include <curand_kernel.h>

__global__ void seedRandomizer(curandState** cs, int xDim, int yDim, int r);

curandState** getCS(int statesX, int statesY);

void freeCS(curandState** cs, int statesX, int statesY);

#endif