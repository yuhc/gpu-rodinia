#ifndef _TESTDATA_GEN_H_
#define _TESTDATA_GEN_H_

#include "parameters.h"

template <typename T>
__inline__ void generateRLETestData(T *data,  unsigned int num_blocks, unsigned int num_block_threads) {
	unsigned int i, j;

	/* generate first block*/
	for(i=0; i<num_block_threads;i+=8) { 
		data[i]=1;  data[i+1]=2; data[i+2]=3;  data[i+3]=3; data[i+4]=3; data[i+5]=4; data[i+6]=4; data[i+7]=5; 
	}
	/*  copy contents of the first block to all other blocks (for testing only)*/
	for(j=1; j<num_blocks;j++)
		for(i=0; i<num_block_threads;i++)
			*(data + j*num_block_threads+i) = data[i];
}

template <typename T>
__inline__ void generateRLETestDataLongRuns1(T *data,  unsigned int num_blocks, unsigned int num_block_threads, unsigned int avg_run_len) {
	unsigned int i, j;

	/* generate first block*/
	for(i=0; i<num_block_threads/avg_run_len;i++)  
		for(j=0; j<avg_run_len;j++)
			data[i*avg_run_len+j]=i; 
	
	/*  copy contents of the first block to all other blocks (for testing only)*/
	for(j=1; j<num_blocks;j++)
		for(i=0; i<num_block_threads;i++)
			*(data + j*num_block_threads+i) = data[i];
}


//VLE TEST DATA VER2.0

//for testing only: generates codewords of the following lengths: 1, 2, 3, 4, 4, 5, 6, 7
// and dummy odewords: 1, 10, 100, 1000, 1000, 10000, 100000, 1000000
// equals 0x01, 0x02, 0x4, 0x8, 0x8, 0x10, 0x20, 0x40
// num_symbols  =256. Must be multiple of 8.
inline void generateCodewords(unsigned int *codewords, unsigned int *codewordlens, unsigned int num_symbols) {
  unsigned int idx, i, j, numbits, k;//val, k;
  /* Generate codeword lengths*/
  for (j=0; j< num_symbols/8; j++) {
    for(i=0; i<4;i++) { //generate first half of length 1,2 3, 4
      idx = j*8 + i;
      codewordlens[idx] = i%4+1;
    }
    for(i=0; i<4;i++) { //generate first half of length 4, 5, 6, 7
      idx = j*8 + 4 + i;
      codewordlens[idx] = i%4 + 4;
    }
  }
  /* Generate codewords*/
  for(k=0; k<num_symbols;k++) { 
    numbits = codewordlens[k];
    codewords[k] = 0x01<<(numbits-1);
  }
}

inline void generateData(unsigned int	*data,  unsigned int num_elements, unsigned int *codewords, unsigned int *codewordlens, unsigned int num_symbols) {
  unsigned int i;
  for(i=0; i<num_elements;i++) { 
	   data[i] = (unsigned int)(((float)rand()/(RAND_MAX + 1)) * num_symbols);
  }
}



#endif