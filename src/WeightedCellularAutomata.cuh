#ifndef WEIGHTEDCELLULARAUTOMATA_CUH
#define WEIGHTEDCELLULARAUTOMATA_CUH

#include <stdio.h>

#include "Empire.cuh"
#include "Queue.cuh"

// The probability to expand over land for each neighbor
#define LANDEXPANSIONPROBABILITY .125
// The probability to expand over water for each neighbor
#define WATEREXPANSIONPROBABILITY .1

float** weightedCellularAutomata(float** map, int width, int height, empire* empires, int numEmpires);

#endif
