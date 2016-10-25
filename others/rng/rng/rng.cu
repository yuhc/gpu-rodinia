/**
* @file rng.cu
* @author Michael Trotter & Matt Goodrum
* @brief Uniform and Normal RNG Implemented in CUDA as device functions
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <sys/time.h>
#define BLOCK_X 16
#define BLOCK_Y 16

/**
* Generates a uniformly distributed number in the range of [0, 1) using the Linear Congruential Generator (LCG)
* @param seed The provided seed array.
* @param index The index within the seed array that will be modified by this function.
* @return A float in the range of [0,1). This float is based on the modified seed value at index in the seed array.
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note GCC's values (M, A, C) for the LCG are used
*/
__device__ float d_randu(int * seed, int index)
{
	
	int M = INT_MAX;
	int A = 1103515245;
	int C = 12345;
	int num = A*seed[index] + C;
	seed[index] = num % M;
	num = seed[index];
	return fabs(num/((float) M));
}

/**
* Generates a normally (Gaussian) distributed number using the Box-Muller transformation
* @param seed The provided seed array.
* @param index The index within the seed array that will be modified as a result of this function
* @return A float in the range of FLOAT_MIN to FLOAT_MAX. This float is based on the modified seed value at index in the seed array.
* @see http://en.wikipedia.org/wiki/Box-Muller_transform
*/
__device__ float d_randn(int * seed, int index){
	//Box-Muller algortihm
	float pi = 3.14159265358979323846;
	float u = d_randu(seed, index);
	float v = d_randu(seed, index);
	float cosine = cos(2*pi*v);
	float rt = -2*log(u);
	return sqrt(rt)*cosine;
}

/**
* A simple main that demonstrates how to setup the seed array for use
*/
int main(){
	//define the length of the seed array
	int length = 10000;
	
	//declare seed array
	int * seed = (int *)malloc(sizeof(int)*length);
	
	//establish original values
	//the current time * the index is good enough for most uses.
	int x;
	for(x = 0; x < length; x++)
	{
		seed[x] = time(0)*x;
	}
	
	//declare GPU copy of seed array
	int * seed_GPU;
	
	//allocate space on GPU
	check_error(cudaMalloc((void **) &seed_GPU, sizeof(int)*length));
	
	//copy seed over
	cudaMemcpy(seed_GPU, seed, sizeof(int)*Nparticles, cudaMemcpyHostToDevice);
	
	//make kernel calls etc; device functions can now use seed array to generate normal and uniform random numbers
	
	//no need to send seed array back 
	
	//free allocated memory
	cudaFree(seed_GPU);
	free(seed);
	
	return 0;
}
