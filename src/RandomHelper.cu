#include "RandomHelper.cuh"
#include <stdio.h>

__global__ void seedRandomizer(curandState** cs, int xDim, int yDim, int r) {
	//Gets the thread numbers
	int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	int threadY = threadIdx.y + blockIdx.y * blockDim.y;

	//Gets the stride
	int strideX = gridDim.x*blockDim.x;
	int strideY = gridDim.y*blockDim.y;

	//Loops through the array seeding the randomizer
	for (int y = threadY; y < yDim; y += strideY) {
		for (int x = threadX; x < xDim; x += strideX) {
			//Seeds the randomizer
			curand_init(1234, r+y + x * xDim, 0, &cs[y][x]);
		}
	}
}

curandState** getCS(int statesX, int statesY) {
	curandState** cs;

	//Allocates the matrix for curandStates
	cudaMallocManaged(&cs, statesY * sizeof(curandState*));

	for (int y = 0; y < statesY; y++) {
		cudaMallocManaged(&cs[y], statesX * sizeof(curandState));
	}

	return cs;
}

void freeCS(curandState** cs, int statesX, int statesY) {
	for (int y = 0; y < statesY; y++) {
		cudaFree(&cs[y]);
	}
	cudaFree(&cs);
}