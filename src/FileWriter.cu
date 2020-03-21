#include "FileWriter.cuh"
#include "Empire.cuh"

void writeFile(char* fileName, float** contents, int width, int height) {
	FILE* f = fopen(fileName, "w");
	fprintf(f, "P3\n%d %d\n255\n\n", width, height);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (contents[y][x]>=0) {
				fprintf(f, "0, %d, 0\t", (int)(contents[y][x]));
			}
			else {
				fprintf(f, "0, 0, %d\t", (int)(contents[y][x]*-1));
			}
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

void writeEmpireFile(char* fileName, empire* empires, float** empireMap, float** map, int width, int height) {
	FILE* f = fopen(fileName, "w");
	fprintf(f, "P3\n%d %d\n255\n\n", width, height);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (map[y][x] >= 0) {
				if (empireMap[y][x]>0) {
					empire emp = empires[int(empireMap[y][x]) - 1];
					fprintf(f, "%d, %d, %d\t", emp.r, emp.b, emp.g);
				}
				else {
					fprintf(f, "255, 255, 255\t");
				}
			}
			else {
				fprintf(f, "0, 0, %d\t", (int)(map[y][x] * -1));
			}
		}
		fprintf(f, "\n");
	}
	fclose(f);
}