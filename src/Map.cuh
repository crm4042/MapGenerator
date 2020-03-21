#ifndef MAP_CUH
#define MAP_CUH

float** generateMap(int width, int height);

void freeMap(float** map, int height);

__global__ void addMapFragments(float** map, float** mapFragment, int width, int height);

#endif