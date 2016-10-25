/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA
 *
 * Copyright (C) 2009 Tjark Bringewat <golvellius@gmx.net>, Ana Balevic <ana.balevic@gmail.com>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * MIT License. Read the full licence: http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home page in your work.
 * 
 */


#ifndef _PACK_KERNELS_H_
#define _PACK_KERNELS_H_
#include "parameters.h"

__global__ static void pack2(unsigned int *srcData, unsigned int *cindex, unsigned int *cindex2, unsigned int *dstData, unsigned int original_num_block_elements) {
	unsigned int tid = blockDim.x*blockIdx.x + threadIdx.x;

	// source index
	unsigned int offset = tid * original_num_block_elements;//DPB,
	unsigned int bitsize = cindex[tid];

	// destination index
	unsigned int pos = cindex2[tid],
				 dword = pos / 32,
				 bit = pos % 32;

	unsigned int i, dw, tmp;
	dw = srcData[offset];			// load the first dword from srcData[]
	tmp = dw >> bit;				// cut off those bits that do not fit into the initial location in destData[]
	atomicOr(&dstData[dword], tmp);	// fill up this initial location
	tmp = dw << 32-bit;				// save the remaining bits that were cut off earlier in tmp
	for (i=1; i<bitsize/32; i++) {	// from now on, we have exclusive access to destData[]
		dw = srcData[offset+i];		// load next dword from srcData[]
		tmp |= dw >> bit;			// fill up tmp
		dstData[dword+i] = tmp;		// write complete dword to destData[]
		tmp = dw << 32-bit;			// save the remaining bits in tmp (like before)
	}
	// exclusive access to dstData[] ends here
	// the remaining block can, or rather should be further optimized
	// write the remaining bits in tmp, UNLESS bit is 0 and bitsize is divisible by 32, in this case do nothing
	if (bit != 0 || bitsize % 32 != 0)
		atomicOr(&dstData[dword+i], tmp);
	if (bitsize % 32 != 0) {
		dw = srcData[offset+i];
		atomicOr(&dstData[dword+i], dw >> bit);
		atomicOr(&dstData[dword+i+1], dw << 32-bit);
	}
}

#endif
