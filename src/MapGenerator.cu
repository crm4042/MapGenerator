#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "EmpireGenerator.cuh"
#include "FileWriter.cuh"
#include "Noise.cuh"
#include "TerrainModifier.cuh"
#include "WeightedCellularAutomata.cuh"

#define DIM 256

void generateMaps() {
	//Seeds the random number generator for the entire program
	srand(time(0));

	printf("Generating the heightmap.\n");

	//Generates the initial map
	float** map = noise(DIM, DIM);

	//Alters the terrain by making oceans below a certain sea level
	submergeTerrain(map, DIM, DIM);

	printf("Generating empire data\n");

	//Generates empires
	char* fileName=(char*)calloc(16, sizeof(char));
	strcpy(fileName, "EmpireNames.txt\0");
	empire* empires=generateEmpires(map, DIM, DIM, fileName);
	free(fileName);

	printf("Generating empire boundaries\n");

	//Uses weighted cellular automata to expand the borders and set an empire capital
	float** empireMap = weightedCellularAutomata(map, DIM, DIM, empires, NUMEMPIRES);

	printf("Writing output\n");

	fileName=(char*)calloc(8, sizeof(char));
	strcpy(fileName, "map.ppm\0");
	writeFile(fileName, map, DIM, DIM);
	free(fileName);

	fileName=(char*)calloc(12, sizeof(char));
	strcpy(fileName, "empires.ppm\0");
	writeEmpireFile(fileName, empires, empireMap, map, DIM, DIM);
	free(fileName);

	printf("Done\n");
}

int main() {
	generateMaps();
	return EXIT_SUCCESS;
}
