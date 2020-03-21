#include "MarkovChain.cuh"

/**
  *  Characters (26)
  *  Start of word (1)
  *  End of word (1)
  */

#define CHARACTERS 27
#define BUFFERSIZE 20
#define START 'S'
#define ENDINDEX 0

int getCharacterIndex(char character) {
	switch (character) {
		//The start character
		case START:
			return CHARACTERS;
		//The end character
		case '\0':
			return ENDINDEX;
		case '\n':
			return ENDINDEX;
		//Any other letter
		default:
			return (int(character) - int('a')+1);
	}

}

void normalizeRow(double** markovMatrix, int row) {
	int sum = 0;
	//Gets the total number of character transitions in the row
	for (int col = 0; col < CHARACTERS; col++) {
		sum+=markovMatrix[row][col];
	}

	//Gets the total number of 
	for (int col = 0; col < CHARACTERS; col++) {
		markovMatrix[row][col]/=sum;
	}
}

double** markovFromFile(char* filename) {
	//Creates a matrix for all characters
	double** markovMatrix=(double**)(calloc(CHARACTERS+1, sizeof(double*)));
	for (int i = 0; i < CHARACTERS+1; i++) {
		markovMatrix[i] = (double*)(calloc(CHARACTERS, sizeof(double)));
	}

	//Opens the file stream
	FILE* f = fopen(filename, "r");

	char* buffer=(char*)(calloc(BUFFERSIZE, sizeof(char)));

	//The current char
	char current = START;

	//Loops through the file to get the next name
	while (fgets(buffer, BUFFERSIZE-1, f)!=NULL) {


		//loops through the buffer
		for (int c = 0; c < BUFFERSIZE-1; c++) {

			//Gets the next character
			char next = buffer[c];
			
			//Changes the next letter to lowercase if it isn't already
			if (next>=int('A') && next<=int('Z')) {
				next = (next-int('A'))+int('a');
			}

			//Adds one to the markov matrix corresponding to the place pointing from the current character to the next character
			markovMatrix[getCharacterIndex(current)][getCharacterIndex(next)]++;

			//Switches the current character
			if (next=='\n' || next== '\0') {
				current = START;
				break;
			}
			else {
				current = next;
			}
		}
	}

	//Loops through the markov matrix and normalizes each row
	for (int r = 0; r < CHARACTERS + 1; r++) {
		normalizeRow(markovMatrix, r);
	}

	return markovMatrix;
}

void freeMarkov(double** markovMatrix) {
	for (int row = 0; row < CHARACTERS + 1; row++) {
		free(markovMatrix[row]);
	}
	free(markovMatrix);
}

char getChar(char currentChar, double** markovMatrix) {
	//Generates random number between 0 and 1
	double randNum = rand()/double(RAND_MAX);

	double sum=0;
	//Loops through the characters to see what range the random number falls in
	for (int i = 0; i < CHARACTERS; i++) {

		//Sums the frequency
		sum += markovMatrix[getCharacterIndex(currentChar)][i];

		//Should return the current character
		if (sum>=randNum) {

			//Terminate the name
			if (currentChar==ENDINDEX) {
				return '\0';
			}

			//Add to the name
			else {
				return char(i + int('a'));
			}
		}
	}
	return '\0';
}

void generateEmpireName(double** markovMatrix, empire* emp) {
	//Creates a name buffer of max length 20
	emp->name = (char*)(calloc(BUFFERSIZE, sizeof(char)));

	char currentChar = START;
	//Loops until the character is eof or until the buffer has no more space
	for (int i = 0; i<BUFFERSIZE-1 && currentChar != '\0'; i++) {
		//Gets the next character
		currentChar = getChar(currentChar, markovMatrix);

		//Adds this to the name
		emp->name[i] = currentChar;
	}
	printf("Empire name=%s\n", emp->name);
}