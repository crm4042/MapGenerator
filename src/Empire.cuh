#ifndef EMPIRE_CUH
#define EMPIRE_CUH

typedef struct emp {
	//The name of the empire
	char* name;
	//The x, y values of the capital city (where the empire started and expanded from)
	int capitalX;
	int capitalY;
	//The number of the empire (how it is represented in the map)
	int number;
	//The rgb values for the color shown on the map
	int r;
	int g;
	int	b;
}empire;

#endif