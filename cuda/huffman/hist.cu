/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include <iostream>
#include <stdio.h>

#define CHECK(ans) {gpuAssert((ans),__FILE__,__LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n",cudaGetErrorString(code),file, line);
        if(abort) exit(code);
    }
}

using namespace std;

#define SIZE    (100*1024*1024)


__global__ void histo_kernel( unsigned char *buffer,
        long size,
        unsigned int *histo ) {

    __shared__  unsigned int temp[256];

    temp[threadIdx.x] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    while (i < size) {
        atomicAdd( &temp[buffer[i]], 1 );
        i += offset;
    }

    __syncthreads();
    atomicAdd( &(histo[threadIdx.x]), temp[threadIdx.x] );
}

int runHisto(char* file, unsigned int* freq, unsigned int memSize, unsigned int *source) {

    FILE *f = fopen(file,"rb");
    if (!f) {perror(file); exit(1);}
    fseek(f,0,SEEK_SET);
    size_t result = fread(source,1,memSize,f);
    if(result != memSize) fputs("Cannot read input file", stderr);

    fclose(f);

    unsigned char *buffer = (unsigned char*)source;

    cudaDeviceProp  prop;
    ( cudaGetDeviceProperties( &prop, 0 ) );
    int blocks = prop.multiProcessorCount;
    if(!prop.deviceOverlap)
    {
        cout << "No overlaps, so no speedup from streams" << endl;
        return 0;
    }

    // allocate memory on the GPU for the file's data
    int partSize = memSize/32;
    int totalNum = memSize/sizeof(unsigned int);
    int partialNum = partSize/sizeof(unsigned int);

    unsigned char *dev_buffer0; 
    unsigned char *dev_buffer1;
    unsigned int *dev_histo;
    cudaMalloc( (void**)&dev_buffer0, partSize ) ;
    cudaMalloc( (void**)&dev_buffer1, partSize ) ;
    cudaMalloc( (void**)&dev_histo,
            256 * sizeof( int ) ) ;
    cudaMemset( dev_histo, 0,
            256 * sizeof( int ) ) ;
    cudaStream_t stream0, stream1;
    CHECK(cudaStreamCreate(&stream0));
    CHECK(cudaStreamCreate(&stream1));
    cudaEvent_t     start, stop;
    ( cudaEventCreate( &start ) );
    ( cudaEventCreate( &stop ) );
    ( cudaEventRecord( start, 0 ) );


    for(int i = 0; i < totalNum; i+=partialNum*2)
    {

        CHECK(cudaMemcpyAsync(dev_buffer0, buffer+i, partSize, cudaMemcpyHostToDevice,stream0));
        CHECK(cudaMemcpyAsync(dev_buffer1, buffer+i+partialNum, partSize, cudaMemcpyHostToDevice,stream1));


        // kernel launch - 2x the number of mps gave best timing
        histo_kernel<<<blocks*2,256,0,stream0>>>( dev_buffer0, partSize, dev_histo );
        histo_kernel<<<blocks*2,256,0,stream1>>>( dev_buffer1, partSize, dev_histo );
    }
    CHECK(cudaStreamSynchronize(stream0));
    CHECK(cudaStreamSynchronize(stream1));
    cudaMemcpy( freq, dev_histo, 256 * sizeof( int ), cudaMemcpyDeviceToHost );
    ( cudaEventRecord( stop, 0 ) );
    ( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    ( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );



    cudaFree( dev_histo );
    cudaFree( dev_buffer0 );
    cudaFree( dev_buffer1 );
    return 0;
}
