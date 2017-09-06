#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include "kmeans.h"

#ifdef WIN
	#include <windows.h>
#else
	#include <pthread.h>
	#include <sys/time.h>
	double gettime() {
		struct timeval t;
		gettimeofday(&t,NULL);
		return t.tv_sec+t.tv_usec*1e-6;
	}
#endif


#ifdef NV 
	#include <oclUtils.h>
#else
	#include <CL/cl.h>
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE RD_WG_SIZE
#else
        #define BLOCK_SIZE 256
#endif

#ifdef RD_WG_SIZE_1_0
     #define BLOCK_SIZE2 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
     #define BLOCK_SIZE2 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
     #define BLOCK_SIZE2 RD_WG_SIZE
#else
     #define BLOCK_SIZE2 256
#endif

//Primitives for timing
#ifdef TIMING
#include "timing.h"

struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_init_start, tv_init_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

int platform_id_inuse = 0;            // platform id in use (default: 0)
int device_id_inuse = 0;              // device id in use (default : 0)

// local variables
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_uint          num_devices;

static int initialize()
{
	cl_int result;
	size_t size;
	cl_uint num_platforms;

	// create OpenCL context
    if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(0,0,*) failed\n"); return -1; }
	cl_platform_id all_platform_id[num_platforms];
	if (clGetPlatformIDs(num_platforms, all_platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(*,*,0) failed\n"); return -1; }
    cl_platform_id platform_id = all_platform_id[platform_id_inuse];

    // get device
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices) != CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };
	printf("num_devices = %d\n", num_devices);
    if (device_id_inuse > num_devices) {
        printf("Invalid Device Number\n");
        return -1;
    }
	device_list = new cl_device_id[num_devices];
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_list, NULL) != CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };

    // get device type
    if (clGetDeviceInfo(device_list[device_id_inuse], CL_DEVICE_TYPE, sizeof(device_type), (void *)&device_type, NULL)!= CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };

	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	if( !context ) { printf("ERROR: clCreateContextFromType(%s) failed\n", device_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU"); return -1; }

	// create command queue for the first device
#ifdef TIMING
	cmd_queue = clCreateCommandQueue( context, device_list[device_id_inuse], CL_QUEUE_PROFILING_ENABLE, NULL );
#else
	cmd_queue = clCreateCommandQueue( context, device_list[device_id_inuse], 0, NULL );
#endif
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }

	return 0;
}

static int shutdown()
{
#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );
	if( device_list ) delete device_list;
#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	// reset all variables
	cmd_queue = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = 0;

	return 0;
}

cl_mem d_feature;
cl_mem d_feature_swap;
cl_mem d_cluster;
cl_mem d_membership;

cl_kernel kernel;
cl_kernel kernel_s;
cl_kernel kernel2;

int   *membership_OCL;
int   *membership_d;
float *feature_d;
float *clusters_d;
float *center_d;

int allocate(int n_points, int n_features, int n_clusters, float **feature)
{

	int sourcesize = 1024*1024;
	char * source = (char *)calloc(sourcesize, sizeof(char)); 
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	// read the kernel core source
	const char * tempchar = "./kmeans.cl";
	FILE * fp = fopen(tempchar, "rb"); 
	if(!fp) { printf("ERROR: unable to open '%s'\n", tempchar); return -1; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);

#ifdef  TIMING
    gettimeofday(&tv_init_start, NULL);
#endif
	// OpenCL initialization
	if(initialize()) return -1;

	// compile kernel
	cl_int err = 0;
	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
	err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
	{ // show warnings/errors
	//	static char log[65536]; memset(log, 0, sizeof(log));
	//	cl_device_id device_id = 0;
	//	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
	//	clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
	//	if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}
	if(err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); return -1; }
	
	const char * kernel_kmeans_c  = "kmeans_kernel_c";
	const char * kernel_swap  = "kmeans_swap";	
		
	kernel_s = clCreateKernel(prog, kernel_kmeans_c, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	kernel2 = clCreateKernel(prog, kernel_swap, &err);  
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
		
	clReleaseProgram(prog);	
#ifdef  TIMING
	gettimeofday(&tv_init_end, NULL);
	tvsub(&tv_init_end, &tv_init_start, &tv);
	init_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif
	
	d_feature = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature (size:%d) => %d\n", n_points * n_features, err); return -1;}
	d_feature_swap = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_feature_swap (size:%d) => %d\n", n_points * n_features, err); return -1;}
	d_cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, n_clusters * n_features  * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_cluster (size:%d) => %d\n", n_clusters * n_features, err); return -1;}
	d_membership = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_membership (size:%d) => %d\n", n_points, err); return -1;}
#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_init_end, &tv);
    mem_alloc_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif
		
	cl_event event;
	//write buffers
	err = clEnqueueWriteBuffer(cmd_queue, d_feature, 1, 0, n_points * n_features * sizeof(float), feature[0], 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_feature (size:%d) => %d\n", n_points * n_features, err); return -1; }
#ifdef TIMING
    h2d_time += probe_event_time(event,cmd_queue);
#endif
    clReleaseEvent(event);

	clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &d_feature);
	clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &d_feature_swap);
	clSetKernelArg(kernel2, 2, sizeof(cl_int), (void*) &n_points);
	clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &n_features);
	
	size_t global_work[3] = { n_points, 1, 1 };
	/// Ke Wang adjustable local group size 2013/08/07 10:37:33
	size_t local_work_size= BLOCK_SIZE; // work group size is defined by RD_WG_SIZE_0 or RD_WG_SIZE_0_0 2014/06/10 17:00:51
	if(global_work[0]%local_work_size !=0)
	  global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;

	err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, &local_work_size, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
#ifdef TIMING
    kernel_time += probe_event_time(event,cmd_queue);
#endif
    clReleaseEvent(event);

	membership_OCL = (int*) malloc(n_points * sizeof(int));
}

void deallocateMemory()
{
#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif
	clReleaseMemObject(d_feature);
	clReleaseMemObject(d_feature_swap);
	clReleaseMemObject(d_cluster);
	clReleaseMemObject(d_membership);
	free(membership_OCL);
#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif
}


int main( int argc, char** argv) 
{
	printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d \n", BLOCK_SIZE, BLOCK_SIZE2);
#ifdef  TIMING
	gettimeofday(&tv_total_start, NULL);
#endif

	setup(argc, argv);
	shutdown();

#ifdef  TIMING
	gettimeofday(&tv_total_end, NULL);
	tvsub(&tv_close_end, &tv_total_start, &tv);
	total_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Init: %f\n", init_time);
	printf("MemAlloc: %f\n", mem_alloc_time);
	printf("HtoD: %f\n", h2d_time);
	printf("Exec: %f\n", kernel_time);
	printf("DtoH: %f\n", d2h_time);
	printf("Close: %f\n", close_time);
	printf("Total: %f\n", total_time);
#endif
}

int	kmeansOCL(float **feature,    /* in: [npoints][nfeatures] */
           int     n_features,
           int     n_points,
           int     n_clusters,
           int    *membership,
		   float **clusters,
		   int     *new_centers_len,
           float  **new_centers)	
{
  
	int delta = 0;
	int i, j, k;
	cl_int err = 0;
	cl_event event;
	
	size_t global_work[3] = { n_points, 1, 1 }; 

	/// Ke Wang adjustable local group size 2013/08/07 10:37:33
	size_t local_work_size=BLOCK_SIZE2; // work group size is defined by RD_WG_SIZE_1 or RD_WG_SIZE_1_0 2014/06/10 17:00:41
	if(global_work[0]%local_work_size !=0)
	  global_work[0]=(global_work[0]/local_work_size+1)*local_work_size;
	
	err = clEnqueueWriteBuffer(cmd_queue, d_cluster, 1, 0, n_clusters * n_features * sizeof(float), clusters[0], 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer d_cluster (size:%d) => %d\n", n_points, err); return -1; }
#ifdef TIMING
    h2d_time += probe_event_time(event,cmd_queue);
#endif
    clReleaseEvent(event);

	int size = 0; int offset = 0;
					
	clSetKernelArg(kernel_s, 0, sizeof(void *), (void*) &d_feature_swap);
	clSetKernelArg(kernel_s, 1, sizeof(void *), (void*) &d_cluster);
	clSetKernelArg(kernel_s, 2, sizeof(void *), (void*) &d_membership);
	clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void*) &n_points);
	clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void*) &n_clusters);
	clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void*) &n_features);
	clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void*) &offset);
	clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void*) &size);

	err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, NULL, global_work, &local_work_size, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
#ifdef TIMING
    h2d_time += probe_event_time(event,cmd_queue);
#endif
    clReleaseEvent(event);

	clFinish(cmd_queue);
	err = clEnqueueReadBuffer(cmd_queue, d_membership, 1, 0, n_points * sizeof(int), membership_OCL, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: Memcopy Out\n"); return -1; }
#ifdef TIMING
    d2h_time += probe_event_time(event,cmd_queue);
#endif
    clReleaseEvent(event);
	
	delta = 0;
	for (i = 0; i < n_points; i++)
	{
		int cluster_id = membership_OCL[i];
		new_centers_len[cluster_id]++;
		if (membership_OCL[i] != membership[i])
		{
			delta++;
			membership[i] = membership_OCL[i];
		}
		for (j = 0; j < n_features; j++)
		{
			new_centers[cluster_id][j] += feature[i][j];
		}
	}

	return delta;
}
