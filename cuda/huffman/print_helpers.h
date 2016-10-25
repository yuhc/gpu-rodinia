#ifndef _PRINT_HELPERS_H_
#define _PRINT_HELPERS_H_

#include "parameters.h"
#include <stdio.h>

__inline void printdbg_data_bin(const char* filename, unsigned int* data, unsigned int num_ints) {
	FILE* dump = fopen((const char*) filename, "wt"); 
	for(unsigned int i=0; i< num_ints; i++) {
		unsigned int mask = 0x80000000;
		for(unsigned int j = 0; j < 32; j++)  {
		if (data[i] & mask) fprintf(dump, "1"); //printf("1");
			else  fprintf(dump, "0");//printf("0");
		mask = mask >> 1;
		}
		fprintf(dump, "\n");
	}
	fclose(dump);
}
__inline void printdbg_data_int(const char* filename, unsigned int* data, unsigned int num_ints) {
	FILE* dump = fopen((const char*) filename, "wt"); 
	for(unsigned int i=0; i< num_ints; i++) {
		fprintf(dump, "%d: %d\n", i, data[i]);
	}
	fclose(dump);
}


__inline void printdbg_gpu_data_detailed(FILE* gpudump, unsigned int* cw32, unsigned int* cw32len, unsigned int* cw32idx, unsigned int num_elements) {
	for(unsigned int i =0; i<num_elements; i++) {
		fprintf(gpudump, "bp: %d, kc: %d, startbit: %d, cwlen: %d, cw:\t\t", cw32idx[i], cw32idx[i]/32, cw32idx[i]%32, cw32len[i]);	
		//print codeword:
		unsigned int mask = 0x80000000;
		mask = mask>>(32-cw32len[i]); 
		for(unsigned int j = 0; j < cw32len[i]; j++)  {
			if (cw32[i] & mask) fprintf(gpudump, "1"); //printf("1");
			else  fprintf(gpudump, "0");//printf("0");
			mask = mask >> 1;
		}
	fprintf(gpudump, "\n");
	}
}


__inline void printdbg_gpu_data_detailed2(const char* filename, unsigned int* cw32, unsigned int* cw32len, unsigned int* cw32idx, unsigned int num_elements) {
	FILE* gpudump = fopen((const char*) filename, "wt"); 
	for(unsigned int i =0; i<num_elements; i++) {
		fprintf(gpudump, "bp: %d, kc: %d, startbit: %d, cwlen: %d, cw:\t\t", cw32idx[i], cw32idx[i]/32, cw32idx[i]%32, cw32len[i]);	
		//print codeword:
		unsigned int mask = 0x80000000;
		mask = mask>>(32-cw32len[i]); 
		for(unsigned int j = 0; j < cw32len[i]; j++)  {
			if (cw32[i] & mask) fprintf(gpudump, "1"); //printf("1");
			else  fprintf(gpudump, "0");//printf("0");
			mask = mask >> 1;
		}
	fprintf(gpudump, "\n");
	}
	fclose(gpudump);
}



/************************************************************************/
/* BIT PRINTS                                                         */
/************************************************************************/
__inline void printBits(unsigned char number) {
	unsigned char mask = 0x80;
	for(unsigned int j = 0; j < 8; j++)  {
		if (number & mask) printf("1");
			else printf("0");
		mask = mask >> 1;
	}
	printf(" ");
}
__inline void print32Bits(unsigned int number) {
	unsigned int mask = 0x80000000;
	for(unsigned int j = 0; j < 32; j++)  {
		if (number & mask) printf("1");
			else printf("0");
		mask = mask >> 1;
	}
	printf("\n");
}
__inline void print32BitsM(unsigned int marker) {
	for(unsigned int j = 0; j < 32; j++)  {
		if (marker==(j+1)) printf("|");
		else printf(".");
	}
	printf("\n");
}
__inline void print_array_char_as_bits(unsigned char* a, unsigned int len) {

	printf(" ========================= Printing vector =======================\n");
	printf("Total number of elements is %d\n", len);
	for(unsigned int i=0; i< len; i++) {
		printf("a[%d]=%d \t", i, a[i]);
		printBits(a[i]);
		printf("\n");

	}
	printf("\n");
	printf(" ==================================================================\n");
}

__inline void print_array_ints_as_bits(unsigned int* a, unsigned int len) {

	printf(" ========================= Printing vector =======================\n");
	for(unsigned int i=0; i< len; i++) {
		print32Bits(a[i]);
		printf("\n");

	}
	printf("\n");
	printf(" ==================================================================\n");
}

__inline void print_compare_array_ints_as_bits(unsigned int* a, unsigned int* b, unsigned int len) {

	printf(" ========================= Printing vector =======================\n");
	for(unsigned int i=0; i< len; i++) {
		print32Bits(a[i]);
		print32Bits(b[i]);
		printf("\n");

	}
	printf("\n");
	printf(" ==================================================================\n");
}


__inline void print_array_in_hex(unsigned int* a, unsigned int len) {

	printf(" ========================= Printing vector =======================\n");
	//printf("Total number of elements is %d\n", len);
	for(unsigned int i=0; i< len; i++) {
		printf("%#X\t", a[i]); 
	}

	printf("\n");
	printf(" ==================================================================\n");
}

/************************************************************************/
/* ARRAY PRINTS                                                        */
/***********************************************************************/

template <typename T>
__inline void print_array(T* a, unsigned int len) {

	printf(" ========================= Printing vector =======================\n");
	printf("Total number of elements is %d\n", len);
	for(unsigned int i=0; i< len; i++) {
		printf("a[%d]=%d \t", i, a[i]);
	}

	printf("\n");
	printf(" ==================================================================\n");
}

template <typename ST, typename CT>
__inline void print_rled_arrays(ST* rle_symbols, CT* rle_counts, unsigned int rle_len) {
	ST current_symbol;
	CT current_count;
	printf(" ========================= Printing RLE vector =======================\n");
	printf(" Total number of RL Pairs is %d\n", rle_len);
	for(unsigned int k = 0; k < rle_len; k++)  {
		current_symbol =  rle_symbols[k];
		current_count =  rle_counts[k];
		printf("(%d,%d) ,\t", current_symbol, current_count);

	}
	printf("\n");
}

__inline void print_packed_rle_array(unsigned int* rle, unsigned int rle_len) {
	unsigned short current_symbol;
	unsigned short current_count;
	printf(" ========================= Printing RLE vector =======================\n");
	printf(" Total number of RL Pairs is %d\n", rle_len);
	for(unsigned int k = 0; k < rle_len; k++)  {
		current_symbol = (unsigned short) (rle[k]>>16);	//get the higher half-word
		current_count =  (unsigned short) rle[k]&0x0000FFFFF;		//get the shorter half-word
		printf("(%d,%d) ,\t", current_symbol, current_count);

	}
	printf("\n");
}

#endif // _PRINT_HELPERS_H_
