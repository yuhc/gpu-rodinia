/*
 * Copyright Ana Balevic, 2008-2009. All rights reserved.
 */
#ifndef _PABIO_KERNEL2_H_
#define _PABIO_KERNEL2_H_

#include "parameters.h"

/* PARALLEL PUT BITS IMPLEMENTATION (CUDA1.1+ compatible)
*  Set numbits in the destination word out[kc] starting from the position startbit
*  Implementation comments:
*  Second atomic operation actually sets these bits to the value stored in the codeword; the other bits are left unotuched
*  First atomic operation is a necessary prepration - we change only the bits that will be affected by the codeword to be written to 1s
*  in order for set bits to work with using atomicand.  
*  TODOs: benchmark performance 1) gm atomics vs sm atomics; 2) memset at init time vs. atomicOr
*/
__device__ void static put_bits_atomic2(unsigned int* out, unsigned int kc,
								unsigned int startbit, unsigned int numbits,
								unsigned int codeword) {
	unsigned int cw32 = codeword;
	unsigned int restbits = 32-startbit-numbits;

	/* 1. Prepare the memory location */
#ifndef MEMSET0 //Can remove this part if the contents of the memory are already set to all 0s
	unsigned int mask = ((1<<numbits)-1);  // -> 0000...001111
	mask<<=restbits;  //fill in zeros at the back positions -> 0000...001111000 -> 11111110000111111111111 (in order not to and other positions)
	atomicAnd(&out[kc], ~mask);		//set 0s in the destination from startbit in the len of numbits
#endif

	/* 2. Write the codeword */
	cw32 = cw32<<restbits;	
	atomicOr(&out[kc], cw32);
}



/* PARALLEL PUT BITS IMPLEMENTATION (CUDA1.1+ compatible)
*  Checkes if the part of the word to be written matches whole memory location, and if yes, avoids using the atmoics.
*  Experience: no benefits, even a bit slower on CUDA.
*/
__device__ void static put_bits_atomic2a(unsigned int* out, unsigned int kc,
								unsigned int startbit, unsigned int numbits,
								unsigned int codeword) {
	unsigned int cw32 = codeword;
	unsigned int restbits = 32-startbit-numbits;

	/* 1. Prepare the memory location */
#ifndef MEMSET0 //Can remove this part if the contents of the memory are already set to all 0s
	unsigned int mask = ((1<<numbits)-1);  // -> 0000...001111
	mask<<=restbits;  //fill in zeros at the back positions -> 0000...001111000 -> 11111110000111111111111 (in order not to and other positions)
	atomicAnd(&out[kc], ~mask);		//set 0s in the destination from startbit in the len of numbits
#endif

	/* 2. Write the codeword */
	if (startbit == 0 && restbits == 0) {
		out[kc] = cw32;
	} else {
		cw32 = cw32<<restbits;	
		atomicOr(&out[kc], cw32);
	}
}
#endif //ifndef _PABIO_KERNEL_H_