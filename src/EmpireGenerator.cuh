#ifndef GENERATEEMPIRES_CUH
#define GENERATEEMPIRES_CUH

#include "MarkovChain.cuh"
#include "Empire.cuh"
//#include "WeightedCellularAutomata.cuh"

#define NUMEMPIRES 10

empire* generateEmpires(float** map, int width, int height, char* filename);

#endif