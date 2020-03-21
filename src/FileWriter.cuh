#ifndef FILEWRITER_CUH
#define FILEWRITER_CUH

#include "Empire.cuh"

#include <stdio.h>
#include <stdlib.h>

void writeFile(char* fileName, float** contents, int width, int height);

void writeEmpireFile(char* fileName, empire* empires, float** empireMap, float** map, int width, int height);

#endif