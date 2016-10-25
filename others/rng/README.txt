Thread-safe CUDA & C/OpenMP Uniform and Normal (Gaussian) Random Number Generator, June 24, 2010

TABLE OF CONTENTS

[1] Introduction
[2] Setting up the seed array
[3] Example Usage
	[3.1] C/OpenMP
	[3.2] CUDA
[4] Description of the functions
	[4.1] randn
	[4.2] randu
	[4.3] d_randn
	[4.4] d_randu
[5] Contact Info

[1] INTRODUCTION

The rng.cu and rng.c files contain the CUDA and C/OpenMP implementations of uniform and normal random number generators (RNGs).
These RNGs are called as functions in the C/OpenMP versions and as device functions in the CUDA version which can be called from within a kernel.
Although these RNGs are thread-safe, they require a seed array that contains a seed for every thread in order to operate.
Examples for setting up the seed array are given in the sample mains inside of the rng files.
Once the seed arrays are set up, one can call the uniform and normal RNGs using the provided functions provided in the rng files.

[2] SETTING UP THE SEED ARRAY

Since the Linear Congruential Generator (LCG) requires large unique integers as seed values, the seeds in the seed array need to be set to such a value.
The current time is a large integer and is unique for a single seed. Since the time does not advance significantly enough while establishing the seed array, 
one can derive a sufficiently unique seed by multiplying the time by the thread index as shown in the sample mains. For the C/OpenMP, this is all that must be
done to begin using the seed array. For the CUDA version, the seed array must be additionally sent to the GPU before it can be called from within a kernel.

[3] EXAMPLE USAGE

The below example usage is from our work on the parallelization of the particle filter. This segment generates a normally distributed number for the x position 
(represented by arrayX) and the y position (represented by arrayY) of each particle that is used to guess where the object will go next.

[3.1] C/OpenMP

#pragma omp parallel for shared(arrayX, arrayY, seed_length, seed) private(x)
for(x = 0; x < seed_length; x++){
	arrayX[x] += 1 + 5*randn(seed, x);
	arrayY[x] += -2 + 2*randn(seed, x);
}

[3.2] CUDA

if(i < seed_length){
	arrayX[i] = arrayX[i] + 1.0 + 5.0*d_randn(seed, i);
	arrayY[i] = arrayY[i] - 2.0 + 2.0*d_randn(seed, i);
	__syncthreads();
}

[4] DESCRIPTION OF THE FUNCTIONS

[4.1] randn

An implementation of the Normal (Gaussian) RNG using the Box-Muller transformation of the uniform distribution in rng.c as a function
Its inputs are the seed array and the index to the specific seed to work with.
Its output is a double representing a normally generated random number.
For more information on the Box-Muller transformation, check out the Wikipedia article on it: http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
For detailed information on the Box-Muller transformation, check out the original paper on it: http://projecteuclid.org/DPubS/Repository/1.0/Disseminate?view=body&id=pdf_1&handle=euclid.aoms/1177706645

[4.2] randu

An implementation of the Uniform RNG using the Linear Congruential Generator (LCG) in rng.c as a function.
It uses the same settings as GCC for its random number generation.
Its inputs are the seed array and the index to the specific seed to work with.
Its output is a double representing a uniformly distributed number on the range [0, 1).
For more information on the LCG, check out the Wikipedia article on it: http://en.wikipedia.org/wiki/Linear_congruential_generator

[4.3] d_randn

An implementation of the Normal (Gaussian) RNG using the Box-Muller transformation of the uniformation distribution in rng.cu as a device function.
Its inputs are the GPU seed array and the index to the specific seed to work with.
Its output is a float representing a normally generated random number.
For more information on the Box-Muller transformation, check out the Wikipedia article on it: http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
For detailed information on the Box-Muller transformation, check out the original paper on it: http://projecteuclid.org/DPubS/Repository/1.0/Disseminate?view=body&id=pdf_1&handle=euclid.aoms/1177706645

[4.4] d_randu

An implementation of the Uniform RNG using the Linear Congruential Generator (LCG) in rng.cu as a device function.
It uses the same settings as GCC for its random number generation.
Its inputs are the GPU seed array and the index to the specific seed to work with.
Its ouput is a float representing a uniformly distributed number on the range [0, 1).
For more information on the LCG, check out the Wikipedia article on it: http://en.wikipedia.org/wiki/Linear_congruential_generator

[5] Contact Info

For questions and additional information about this RNG, please contact Michael Trotter (mjt5v@virginia.edu) or Matt Goodrum (mag6x@virginia.edu).