#ifndef MARKOVCHAIN_CUH
#define MARKOVCHAIN_CUH

#include "Empire.cuh"

#include <stdlib.h>
#include <stdio.h>

double** markovFromFile(char* filename);

void freeMarkov(double** markovMatrix);

void generateEmpireName(double** markovMatrix, empire* emp);

#endif