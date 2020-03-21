#ifndef TERRAINMODIFIER_CUH
#define TERRAINMODIFIER_CUH

#include <stdio.h>

#define MAXBLOCKS 8
#define BLOCKSIZE 256
#define WATERLEVEL 128

void submergeTerrain(float** map, int width, int height);

#endif