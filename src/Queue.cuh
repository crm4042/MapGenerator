#ifndef QUEUE_CUH
#define QUEUE_CUH

typedef struct q{
	int height;
	int width;
	int** queue;
	int entries;
}Queue;

Queue** createQueue(int width, int height);

void freeQueue(Queue** queue);

void freeEntryFromQueue(int* entry);

int isEmpty(Queue** queue);

void insertToQueue(Queue** queue, int* entry);

int* peekFromQueue(Queue** queue, int entryNum);

int* popFromQueue(Queue** queue, int entryNum);

int* popRandomEntryFromQueue(Queue** queue);

void printQueueData(Queue** queue);

#endif
