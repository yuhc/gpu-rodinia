/**
* @file rng.c
* @author Michael Trotter & Matt Goodrum
* @brief Uniform and Normal RNG Implemented in OpenMP
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <limits.h>
#define PI acos(-1)
/**
@var M value for Linear Congruential Generator (LCG); use GCC's value
*/
long M = INT_MAX;
/**
@var A value for LCG
*/
int A = 1103515245;
/**
@var C value for LCG
*/
int C = 12345;
/**
* Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
* @see http://en.wikipedia.org/wiki/Linear_congruential_generator
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a uniformly distributed number [0, 1)
*/
double randu(int * seed, int index)
{
	int num = A*seed[index] + C;
	seed[index] = num % M;
	return fabs(seed[index]/((double) M));
}
/**
* Generates a normally distributed random number using the Box-Muller transformation
* @note This function is thread-safe
* @param seed The seed array
* @param index The specific index of the seed to be advanced
* @return a double representing random number generated using the Box-Muller algorithm
* @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
*/
double randn(int * seed, int index){
	/*Box-Muller algorithm*/
	double u = randu(seed, index);
	double v = randu(seed, index);
	double cosine = cos(2*PI*v);
	double rt = -2*log(u);
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
	
	//make kernel calls etc; device functions can now use seed array to generate normal and uniform random numbers
	/* Example
	
		#pragma omp parallel for shared(arrayX, arrayY, length, seed) private(x)
		for(x = 0; x < length; x++){
			arrayX[x] += 1 + 5*randn(seed, x);
			arrayY[x] += -2 + 2*randn(seed, x);
		}
	*/
	
	//free allocated memory
	free(seed);
	
	return 0;
}