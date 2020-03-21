#include "WeightedCellularAutomata.cuh"

/**
  *	Creates a queue entry of x and y values
  *	Parameter x: the x value
  *	Parameter y: the y value
  *	Returns: an entry to insert into the queue
  */

int* createQueueEntry(int x, int y){
	int* entry=(int*)calloc(2, sizeof(int));
	entry[0]=x;
	entry[1]=y;
	return entry;
}

void addSurroundingsToQueue(Queue** queue, float** empireMap, int width, int height, int x, int y){
	for(int surroundingY=max(0, y-1); surroundingY<min(height, y+2); surroundingY++){
		for(int surroundingX=max(0, x-1); surroundingX<min(width, x+2); surroundingX++){
			if(empireMap[surroundingY][surroundingX]==0 && (x!=surroundingX || y!=surroundingY)){
				insertToQueue(queue, createQueueEntry(surroundingX, surroundingY));
			}
		}
	}
}

/**
  *	Generates the capitals (starting points) for each civilization
  *	Each civilization will expand outwards from there
  *	Parameter queue: the queue to add the surrounding pixels to
  *	Parameter empireMap: the map of empires
  *	Parameter map: the map of land and water
  *	Parameter width: the width of the maps
  *	Parameter height: the height of the maps
  *	Parameter empires: the list of empires to draw
  *	Parameter numEmpires: the number of empires in the list
  *	Returns: nothing
  */

void generateCapitals(Queue** queue, float** empireMap, float** map, int width, int height, empire* empires, int numEmpires){
	for(int empire=0; empire<numEmpires; empire++){
		
		int capitalX;
		int capitalY;
		
		do{
			capitalX=(int)((((double)rand())/RAND_MAX)*width);
			capitalY=(int)((((double)rand())/RAND_MAX)*height);
		}while(map[capitalY][capitalX]<0);

		empires[empire].capitalX=capitalX;
		empires[empire].capitalY=capitalY;

		empireMap[capitalY][capitalX]=empires[empire].number;

		addSurroundingsToQueue(queue, empireMap, width, height, capitalX, capitalY);
	}
}

/**
  *	Checks a single entry from the queue for possible expansions
  *	Parameter queue: the queue to get the tile from
  *	Parameter empireMap: the map of empires to possibly add the tile to
  *	Parameter map: the map of land/sea height values
  *	Parameter width: the width of the maps
  *	Parameter height: the height of the maps
  *	Parameter empires: the list of empires
  *	Parameter numEmpires: the number of empires in the list
  */

void checkTile(Queue** queue, float** empireMap, float** map, int width, int height, empire* empires, int numEmpires){
	
	int* entry=popRandomEntryFromQueue(queue);

	if(empireMap[entry[1]][entry[0]]!=0){
		free(entry);
		return;
	}

	int* surroundingTilesForEmpire=(int*)calloc(numEmpires, sizeof(int));

	// Counts the number of surrounding tiles for each empire
	for(int y=max(0, entry[1]-1); y<min(height, entry[1]+2); y++){
		for(int x=max(0, entry[0]-1); x<min(width, entry[0]+2); x++){
			if(empireMap[y][x]>0){
				surroundingTilesForEmpire[((int)empireMap[y][x])-1]++;
			}
		}
	}

	float randomProbability=((float)rand())/(RAND_MAX+1);
	int isLand=map[entry[1]][entry[0]]>0;

	// Gets the empire assoicated with the random probability
	for(int empire=0; empire<numEmpires; empire++){
		if(isLand){
			randomProbability-=surroundingTilesForEmpire[empire]*LANDEXPANSIONPROBABILITY;
		}
		else{
			randomProbability-=surroundingTilesForEmpire[empire]*WATEREXPANSIONPROBABILITY;
		}

		if(randomProbability<0){
			empireMap[entry[1]][entry[0]]=empire+1;
			break;
		}
	}

	if(randomProbability>=0){
		insertToQueue(queue, entry);
	}
	else{
		addSurroundingsToQueue(queue, empireMap, width, height, entry[0], entry[1]);
	}

	free(surroundingTilesForEmpire);
}

/**
  *	Creates a map for empires based on a series of probabalistic expansions
  *	Parameter map: the map of height values for water/land
  *	Parameter width: the width of the map
  *	Parameter height: the height of the map
  *	Parameter empires: the list of empires
  *	Parameter numEmpires: the number of empires in the list
  *	Returns: the map of empires with their corresponding numbers
  */

float** weightedCellularAutomata(float** map, int width, int height, empire* empires, int numEmpires){
	// Creates a 10x2 queue
	Queue** queue=createQueue(2, 10);

	// Creats the empire map
	float** empireMap=(float**)calloc(height, sizeof(float*));
	for(int y=0; y<height; y++){
		empireMap[y]=(float*)calloc(width, sizeof(float));
	}

	// Creates the capitals and adds the surrounding tiles to the queue
	generateCapitals(queue, empireMap, map, width, height, empires, numEmpires);

	printf("Capitals made %d for %d empires\n", (*queue)->entries, numEmpires);

	// Expands the empires
	while(!isEmpty(queue)){
		checkTile(queue, empireMap, map, width, height, empires, numEmpires);
	}

	freeQueue(queue);

	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){
			if(empireMap[y][x]==0){
				printf("Here\n");
			}
		}
	}

	return empireMap;
}
