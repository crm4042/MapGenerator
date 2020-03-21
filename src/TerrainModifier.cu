#include "TerrainModifier.cuh"

__global__ void submerge(float** map, int width, int height){
	//Gets the thread numbers
	int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	int threadY = threadIdx.y + blockIdx.y * blockDim.y;

	//Gets the stride to increase
	int strideX = gridDim.x*blockDim.x;
	int strideY = gridDim.y*blockDim.y;

	for (int y = threadY; y < height; y+=strideY) {
		for (int x = threadX; x < width; x+=strideX) {
			if (map[y][x]<=WATERLEVEL) {
				map[y][x] *= -1;
			}
		}
	}
}

void submergeTerrain(float** map, int width, int height) {
	submerge<<<dim3(BLOCKSIZE, BLOCKSIZE), dim3(MAXBLOCKS, MAXBLOCKS)>>>(map, width, height);
	cudaDeviceSynchronize();
}