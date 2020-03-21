#include "Map.cuh"

float** generateMap(int width, int height) {
	float** map;
	cudaMallocManaged(&map, height*sizeof(float*));
	for (int y = 0; y < height; y++) {
		cudaMallocManaged(&map[y], width * sizeof(float));
	}
	return map;
}

void freeMap(float** map, int height) {
	for (int y = 0; y < height; y++) {
		cudaFree(&map[y]);
	}
	cudaFree(&map);
}

__global__ void addMapFragments(float** map, float** mapFragment, int width, int height) {
	//Gets the thread numbers
	int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	int threadY = threadIdx.y + blockIdx.y * blockDim.y;

	//Gets the stride to increase
	int strideX = gridDim.x*blockDim.x;
	int strideY = gridDim.y*blockDim.y;
	for (int y = threadY; y < height; y+=strideY) {
		for (int x = threadX; x < width; x+=strideX) {
			map[y][x] += mapFragment[y][x];
		}
	}
}