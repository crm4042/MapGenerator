#include <stdio.h>
#include "Queue.cuh"

Queue** createQueue(int width, int height){
	Queue** queue=(Queue**)calloc(1, sizeof(Queue*));
	
	*queue=(Queue*)calloc(1, sizeof(Queue));

	(*queue)->width=width;
	(*queue)->height=height;
	(*queue)->entries=0;

	(*queue)->queue=(int**)calloc(height, sizeof(int*));
	for(int entry=0; entry<height; entry++){
		(*queue)->queue[entry]=(int*)calloc(width, sizeof(int));
	}

	return queue;
}

void copyValuesBetweenQueues(Queue** queue1, Queue** queue2){
	for(int entry=0; entry<(*queue1)->height; entry++){
		for(int value=0; value<(*queue1)->width; value++){
			(*queue2)->queue[entry][value]=(*queue1)->queue[entry][value];
		}
	}

	(*queue2)->entries=(*queue1)->entries;
}

void freeQueue(Queue** queue){
	for(int entry=0; entry<(*queue)->height; entry++){
		free((*queue)->queue[entry]);
	}
	free((*queue)->queue);
	free(*queue);
	free(queue);
}

void extendQueue(Queue** queue, int newWidth, int newHeight){
	Queue** queueHolder=createQueue(newWidth, newHeight);
	copyValuesBetweenQueues(queue, queueHolder);
	*queue=*queueHolder;
	free(queueHolder);
}

void freeEntryFromQueue(int* entry){
	free(entry);
}

int isEmpty(Queue** queue){
	return (*queue)->entries==0;
}

void insertToQueue(Queue** queue, int* entry){
	if((*queue)->entries==(*queue)->height){
		extendQueue(queue, (*queue)->width, (*queue)->height*2);
	}

	for(int value=0; value<(*queue)->width; value++){
		(*queue)->queue[(*queue)->entries][value]=entry[value];
	}
	(*queue)->entries++;

	freeEntryFromQueue(entry);
}

int* peekFromQueue(Queue** queue, int entryNum){
	return (*queue)->queue[entryNum];
}

int* popFromQueue(Queue** queue, int entryNum){
	int* returnedEntry=(int*)calloc((*queue)->width, sizeof(int));
	for(int value=0; value<(*queue)->width; value++){
		returnedEntry[value]=(*queue)->queue[entryNum][value];
	}
	
	(*queue)->entries--;
	for(int entry=entryNum; entry<(*queue)->entries; entry++){
		for(int value=0; value<(*queue)->width; value++){
			(*queue)->queue[entry][value]=(*queue)->queue[entry+1][value];
		}
	}

	return returnedEntry;
}

int* popRandomEntryFromQueue(Queue** queue){
	int entry=(int)((((double)rand())/(RAND_MAX+1))*(*queue)->entries);
	return popFromQueue(queue, entry);
}

void printQueueData(Queue** queue){
	printf("Width %d, Height %d, entries %d\n", (*queue)->width, (*queue)->height, (*queue)->entries);

	for(int entry=0; entry<(*queue)->height; entry++){
		printf("Entry %d \t", entry);
		for(int value=0; value<(*queue)->width; value++){
			printf("%d ", (*queue)->queue[entry][value]);
		}
		printf("\n");
	}
}
