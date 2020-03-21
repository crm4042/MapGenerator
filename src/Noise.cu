#include "Noise.cuh"

__global__ void noiseKernel(float** mapFragment, int delta, int width, int height, curandState** cs) {
	//Gets the thread numbers
	int threadX = threadIdx.x + blockIdx.x * blockDim.x;
	int threadY = threadIdx.y + blockIdx.y * blockDim.y;

	//Gets the stride to increase
	int strideX = gridDim.x*blockDim.x;
	int strideY = gridDim.y*blockDim.y;

	//Loops through the blocks and generates random numbers on the corners
	for (int y = threadY * delta; y < height; y += strideY * delta) {
		for (int x = threadX * delta; x < width; x += strideX * delta) {
			//The y-bound and weight for the weighted random number
			int yBound = y + delta;
			float weightedY = 1;
			if (yBound >= height) {
				yBound = height - 1;
				weightedY = 1 - (y*1.0 + delta - yBound) / delta;
			}

			//The x-bound and weight for the weighted random number
			int xBound = x + delta;
			float weightedX = 1;
			if (xBound >= width) {
				xBound = width - 1;
				weightedX = 1 - (x*1.0 + delta - xBound) / delta;
			}

			//Generate top left random value
			if (x == 0 && y == 0) {
				mapFragment[y][x] = curand_uniform(&cs[y / delta][x / delta])*delta;
			}

			//Generate top right random value
			if (y == 0) {
				mapFragment[y][xBound] = curand_uniform(&cs[y / delta][x / delta + 1])*weightedX*delta;
			}

			//Generate bottom left random value
			if (x == 0) {
				mapFragment[yBound][x] = curand_uniform(&cs[y / delta + 1][x / delta])*weightedY*delta;
			}

			//Generate bottom right random value
			mapFragment[yBound][xBound] = curand_uniform(&cs[y / delta+1][x / delta+1])*weightedX*weightedY*delta;

			//Interpolates values
			for (int interpolateY = y; interpolateY <= yBound; interpolateY++) {
				//Fraction of the distance to the y-Bound
				float fraction = (interpolateY*1.0 - y) / (yBound - y);
				//Interpolates the two y-values
				if (interpolateY!=y && interpolateY!=yBound) {
					//Uses cossine interpolation to get a smooth gradient
					fraction = (1 - cospif(fraction))/2;
					mapFragment[interpolateY][x] = (1 - fraction)*mapFragment[y][x] + fraction * mapFragment[yBound][x];
					mapFragment[interpolateY][xBound] = (1 - fraction)*mapFragment[y][xBound] + fraction * mapFragment[yBound][xBound];
				}

				//Loops through all x's and interpolates them
				for (int interpolateX = x+1; interpolateX <= xBound-1; interpolateX++) {
					//Fraction of the distance to the x-Bound
					fraction = (interpolateX*1.0 - x) / (xBound - x);
					//Uses cossine interpolation to get a smooth gradient
					fraction = (1-cospif(fraction))/2;
					mapFragment[interpolateY][interpolateX] = (1 - fraction)*mapFragment[interpolateY][x] + fraction * mapFragment[interpolateY][xBound];
				}
			}
		}
	}
}

float** noise(int width, int height) {

	//Generates the map
	float** map = generateMap(width, height);

	for (int i = 7; i >= 0; i--) {
		int delta = pow(2, i);

		//Allocates the matrix for curandStates
		int statesY = height / delta + 2;
		int statesX = width / delta + 2;
		curandState** cs=getCS(statesY, statesX);

		//Initializes the seeds in parallel
		seedRandomizer << <dim3(BLOCKSIZE, BLOCKSIZE), dim3(MAXBLOCKS, MAXBLOCKS) >> >(cs, width / delta + 2, height / delta + 2, rand());

		cudaDeviceSynchronize();

		float** mapFragment = generateMap(width, height);

		//Uses the noise kernel
		noiseKernel << <dim3(BLOCKSIZE, BLOCKSIZE), dim3(MAXBLOCKS, MAXBLOCKS) >> >(mapFragment, delta, width, height, cs);

		cudaDeviceSynchronize();

		//Frees the curandStates
		freeCS(cs, statesX, statesY);

		//Adds the map fragment to the map
		addMapFragments << <dim3(BLOCKSIZE, BLOCKSIZE), dim3(MAXBLOCKS, MAXBLOCKS) >> >(map, mapFragment, width, height);

		freeMap(mapFragment, height);

		cudaDeviceSynchronize();
	}
	return map;
}
