#ifndef _PARAMS_H_
#define _PARAMS_H_

typedef unsigned int uint;
typedef unsigned char uint8;

#define BENCH 0 
/* 0 - MEASURE TIME, NO TESTING
** 1 - TEST
** 2 - TEST & VERBOSE 
*/
#define TESTING

#define DPT 4               // data (dwords) per thread

#define CACHECWLUT			// MAX DPT = 8
//#define CACHESRCDATA		// MAX DPT = 4

#define SMATOMICS

#define MEMSET0

#define MAX_SM_BLOCK_SIZE_GPU 16384 //B

#define NUM_SYMBOLS 256 //fixed to 256.

#endif