#include "EmpireGenerator.cuh"

int generateRandomColor() {
	return rand() % 256;
}

void generateEmpire(empire* emp, int empNum, double** markovMatrix) {
	//Generates the empire's number
	emp->number = empNum;
	
	//Generates the empire's name
	generateEmpireName(markovMatrix, emp);

	//Generates the color for the empire
	emp->r = generateRandomColor();
	emp->g = generateRandomColor();
	emp->b = generateRandomColor();
}

empire* generateEmpires(float** map, int width, int height, char* filename) {
	//Allocates the empires for GPU processing
	empire* empires;
	cudaMallocManaged(&empires, NUMEMPIRES*sizeof(empire));
	
	//The matrix for the markov chain
	double** markovMatrix = markovFromFile(filename);

	//Generates all the empire structs and their names
	for (int i = 0; i < NUMEMPIRES; i++) {
		//The number is 1 more than i reaching from [1-NUMEMPIRES]
		generateEmpire(&empires[i], i+1, markovMatrix);
	}

	freeMarkov(markovMatrix);

	return empires;
}
