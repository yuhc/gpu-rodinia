#ifndef __GAUSSIAN_ELIMINATION__
#define __GAUSSIAN_ELIMINATION__

#include "gaussianElim.h"
#include <math.h>

#include "timing.h"

#ifdef RD_WG_SIZE_0_0
        #define BLOCK_SIZE_0 RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define BLOCK_SIZE_0 RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_0 RD_WG_SIZE
#else
        #define BLOCK_SIZE_0 0
#endif

//2D defines. Go from specific to general
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_1_X RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_1_X RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_1_X RD_WG_SIZE
#else
        #define BLOCK_SIZE_1_X 0
#endif

#ifdef RD_WG_SIZE_1_1
        #define BLOCK_SIZE_1_Y RD_WG_SIZE_1_1
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_1_Y RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_1_Y RD_WG_SIZE
#else
        #define BLOCK_SIZE_1_Y 0
#endif

//Primitives for timing
#ifdef TIMING
struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_init_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time= 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

cl_context context=NULL;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }


  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }


}


int main(int argc, char *argv[]) {

  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", BLOCK_SIZE_0, BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y);
    float *a=NULL, *b=NULL, *finalVec=NULL;
    float *m=NULL;
    int size = -1;
    
    FILE *fp;
    
    // args
    char filename[200];
    int show_data=0,quiet=0,timing=0,platform=0,device=0;
    
    // parse command line
    if (parseCommandline(argc, argv, filename,
			 &quiet, &show_data, &timing, &platform, &device, &size)) {
    printUsage();
    return 0;
    }

#ifdef  TIMING
    gettimeofday(&tv_total_start, NULL);
#endif
    context = cl_init_context(platform,device,quiet);
#ifdef  TIMING
	gettimeofday(&tv_init_end, NULL);
	tvsub(&tv_init_end, &tv_total_start, &tv);
	init_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    if(size < 1)
    {
	fp = fopen(filename, "r");
	fscanf(fp, "%d", &size);

	a = (float *) malloc(size * size * sizeof(float));
	InitMat(fp,size, a, size, size);

	b = (float *) malloc(size * sizeof(float));
	InitAry(fp, b, size);

	fclose(fp);

      }
    else
      {
	printf("create input internally before create, size = %d \n", size);

	a = (float *) malloc(size * size * sizeof(float));
	create_matrix(a, size);

	b = (float *) malloc(size * sizeof(float));
	for (int i =0; i< size; i++)
	  b[i]=1.0;

      }

    if (!quiet && show_data) {
      printf("The input matrix a is:\n");
      PrintMat(a, size, size, size);

      printf("The input array b is:\n");
      PrintAry(b, size);
    }
 
    // create the solution matrix
    m = (float *) malloc(size * size * sizeof(float));
	 
    // create a new vector to hold the final answer

    finalVec = (float *) malloc(size * sizeof(float));
    
    InitPerRun(size,m);

    // run kernels
	ForwardSub(context,a,b,m,size,timing);

    if (!quiet && show_data) {
        printf("The result of matrix m is: \n");

        PrintMat(m, size, size, size);
        printf("The result of matrix a is: \n");
        PrintMat(a, size, size, size);
        printf("The result of array b is: \n");
        PrintAry(b, size);

        BackSub(a,b,finalVec,size);
        printf("The final solution is: \n");
        PrintAry(finalVec,size);
    }

    free(m);
    free(a);
    free(b);
    free(finalVec);

#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif

    cl_cleanup();

#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
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
  //OpenClGaussianElimination(context,timing);

  return 0;
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub(cl_context context, float *a, float *b, float *m, int size,int timing){    
    // 1. set up kernels
    cl_kernel fan1_kernel,fan2_kernel;
    cl_int status=0;
    cl_program gaussianElim_program;
    cl_event writeEvent,kernelEvent,readEvent;
    float writeMB=0,readMB=0;

    gaussianElim_program = cl_compileProgram(
        (char *)"gaussianElim_kernels.cl",NULL);

    fan1_kernel = clCreateKernel(
        gaussianElim_program, "Fan1", &status);
    status = cl_errChk(status, (char *)"Error Creating Fan1 kernel",true);
    if(status)exit(1);
   
    fan2_kernel = clCreateKernel(
        gaussianElim_program, "Fan2", &status);
    status = cl_errChk(status, (char *)"Error Creating Fan2 kernel",true);
    if(status)exit(1);
    
    // 2. set up memory on device and send ipts data to device

    cl_mem a_dev, b_dev, m_dev;

    cl_int error=0;

#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_start, NULL);
#endif
    a_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float)*size*size, NULL, &error);
    b_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float)*size, NULL, &error);
    m_dev = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * size * size, NULL, &error);
#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_mem_alloc_start, &tv);
    mem_alloc_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

    cl_command_queue command_queue = cl_getCommandQueue();

    error = clEnqueueWriteBuffer(command_queue,
               a_dev,
               1, // change to 0 for nonblocking write
               0, // offset
               sizeof(float)*size*size,
               a,
               0,
               NULL,
               &writeEvent);
#ifdef TIMING
    h2d_time += eventTime(writeEvent,command_queue);
#endif
    clReleaseEvent(writeEvent);

    error = clEnqueueWriteBuffer(command_queue,
               b_dev,
               1, // change to 0 for nonblocking write
               0, // offset
               sizeof(float)*size,
               b,
               0,
               NULL,
               &writeEvent);
#ifdef TIMING
    h2d_time += eventTime(writeEvent,command_queue);
#endif
    clReleaseEvent(writeEvent);

    error = clEnqueueWriteBuffer(command_queue,
               m_dev,
               1, // change to 0 for nonblocking write
               0, // offset
               sizeof(float)*size*size,
               m,
               0,
               NULL,
               &writeEvent);
#ifdef TIMING
    h2d_time += eventTime(writeEvent,command_queue);
#endif
    clReleaseEvent(writeEvent);

    writeMB = (float)(sizeof(float) * size * (size + size + 1) / 1e6);

    // 3. Determine block sizes
    size_t globalWorksizeFan1[1];
    size_t globalWorksizeFan2[2];
    size_t localWorksizeFan1Buf[1]={BLOCK_SIZE_0};
    size_t localWorksizeFan2Buf[2]={BLOCK_SIZE_1_X, BLOCK_SIZE_1_Y};
    size_t *localWorksizeFan1=NULL;
    size_t *localWorksizeFan2=NULL;

        globalWorksizeFan1[0] = size;
        globalWorksizeFan2[0] = size;
        globalWorksizeFan2[1] = size;

        if(localWorksizeFan1Buf[0]){
                localWorksizeFan1=localWorksizeFan1Buf;
                globalWorksizeFan1[0]=(int)ceil(globalWorksizeFan1[0]/(double)localWorks\
izeFan1Buf[0])*localWorksizeFan1Buf[0];
        }
        if(localWorksizeFan2Buf[0]){
                localWorksizeFan2=localWorksizeFan2Buf;
                globalWorksizeFan2[0]=(int)ceil(globalWorksizeFan2[0]/(double)localWorks\
izeFan2Buf[0])*localWorksizeFan2Buf[0];
                globalWorksizeFan2[1]=(int)ceil(globalWorksizeFan2[1]/(double)localWorks\
izeFan2Buf[1])*localWorksizeFan2Buf[1];
        }

	int t;
	// 4. Setup and Run kernels
	for (t=0; t<(size-1); t++) {
        // kernel args
        cl_int argchk;
        argchk  = clSetKernelArg(fan1_kernel, 0, sizeof(cl_mem), (void *)&m_dev);
        argchk |= clSetKernelArg(fan1_kernel, 1, sizeof(cl_mem), (void *)&a_dev);
        argchk |= clSetKernelArg(fan1_kernel, 2, sizeof(cl_mem), (void *)&b_dev);
        argchk |= clSetKernelArg(fan1_kernel, 3, sizeof(int), (void *)&size);
        argchk |= clSetKernelArg(fan1_kernel, 4, sizeof(int), (void *)&t);
    
        cl_errChk(argchk,"ERROR in Setting Fan1 kernel args",true);

        // launch kernel
        error = clEnqueueNDRangeKernel(
                  command_queue,  fan1_kernel, 1, 0,
                  globalWorksizeFan1,localWorksizeFan1,
                  0, NULL, &kernelEvent);

        cl_errChk(error,"ERROR in Executing Fan1 Kernel",true);
#ifdef TIMING
        kernel_time += eventTime(kernelEvent,command_queue);
#endif
        clReleaseEvent(kernelEvent);

		// kernel args
		argchk  = clSetKernelArg(fan2_kernel, 0, sizeof(cl_mem), (void *)&m_dev);
        argchk |= clSetKernelArg(fan2_kernel, 1, sizeof(cl_mem), (void *)&a_dev);
        argchk |= clSetKernelArg(fan2_kernel, 2, sizeof(cl_mem), (void *)&b_dev);
        argchk |= clSetKernelArg(fan2_kernel, 3, sizeof(int), (void *)&size);
        argchk |= clSetKernelArg(fan2_kernel, 4, sizeof(int), (void *)&t);

        cl_errChk(argchk,"ERROR in Setting Fan2 kernel args",true);

        // launch kernel
        error = clEnqueueNDRangeKernel(
                  command_queue,  fan2_kernel, 2, 0,
                  globalWorksizeFan2,NULL,
                  0, NULL, &kernelEvent);

        cl_errChk(error,"ERROR in Executing Fan1 Kernel",true);
#ifdef TIMING
        kernel_time+=eventTime(kernelEvent,command_queue);
#endif
        clReleaseEvent(kernelEvent);
	}

    // 5. transfer data off of device
    error = clEnqueueReadBuffer(command_queue,
        a_dev,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * size * size,
        a,
        0,
        NULL,
        &readEvent);

    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
#ifdef TIMING
    d2h_time +=eventTime(readEvent,command_queue);
#endif
    clReleaseEvent(readEvent);

    error = clEnqueueReadBuffer(command_queue,
        b_dev,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * size,
        b,
        0,
        NULL,
        &readEvent);
    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
#ifdef TIMING
    d2h_time +=eventTime(readEvent,command_queue);
#endif
    clReleaseEvent(readEvent);

    error = clEnqueueReadBuffer(command_queue,
        m_dev,
        1, // change to 0 for nonblocking write
        0, // offset
        sizeof(float) * size * size,
        m,
        0,
        NULL,
        &readEvent);

    cl_errChk(error,"ERROR with clEnqueueReadBuffer",true);
#ifdef TIMING
    d2h_time +=eventTime(readEvent,command_queue);
#endif
    clReleaseEvent(readEvent);

    readMB = (float)(sizeof(float) * size * (size + size + 1) / 1e6);
clFinish(command_queue);

#ifdef TIMING
    printf("Matrix Size\tWrite(s) [size]\t\tKernel(s)\tRead(s)  [size]\t\tTotal(s)\n");
    printf("%dx%d      \t",size,size);
    printf("%f [%.2fMB]\t",h2d_time,writeMB);
    printf("%f\t",kernel_time);
    printf("%f [%.2fMB]\t",d2h_time,readMB);
    printf("%f\n\n",h2d_time+kernel_time+d2h_time);
#endif

#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif
    clReleaseMemObject(a_dev);
    clReleaseMemObject(b_dev);
    clReleaseMemObject(m_dev);
#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif
}

float eventTime(cl_event event,cl_command_queue command_queue){
    cl_int error=0;
    cl_ulong eventStart,eventEnd;
    clFinish(command_queue);
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,
                                    sizeof(cl_ulong),&eventStart,NULL);
    cl_errChk(error,"ERROR in Event Profiling.",true); 
    error = clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong),&eventEnd,NULL);
    cl_errChk(error,"ERROR in Event Profiling.",true);

    return (float)((eventEnd-eventStart)/1000000.0);
}

 // Ke Wang add a function to generate input internally
int parseCommandline(int argc, char *argv[], char* filename,
                     int *q, int *v, int *t, int *p, int *d, int *size){
    int i;
    if (argc < 2) return 1; // error
    char flag;

    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 's': // matrix size
              i++;
              *size = atoi(argv[i]);
	      printf("Create matrix internally in parse, size = %d \n", *size);
              break;
            case 'f': // matrix file
              i++;
	      strncpy(filename,argv[i],100);
	      printf("Read file from %s \n", filename);
              break;
            case 'h': // help
              return 1;
              break;
            case 'q': // quiet
              *q = 1;
              break;
			case 'v': // show_data
			  *v = 1;
			  break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Gaussian Elimination Usage\n");
  printf("\n");
  printf("gaussianElimination [filename] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./gaussianElimination matrix4.txt\n");
  printf("\n");
  printf("filename     the filename that holds the matrix data\n");
  printf("\n");
  printf("-h           Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}


/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun(int size,float *m) 
{
	int i;
	for (i=0; i<size*size; i++)
			*(m+i) = 0.0;
}
void BackSub(float *a, float *b, float *finalVec, int size)
{
	// solve "bottom up"
	int i,j;
	for(i=0;i<size;i++){
		finalVec[size-i-1]=b[size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[size-i-1]-=*(a+size*(size-i-1)+(size-j-1)) * finalVec[size-j-1];
		}
		finalVec[size-i-1]=finalVec[size-i-1]/ *(a+size*(size-i-1)+(size-i-1));
	}
}
void InitMat(FILE *fp, int size, float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			fscanf(fp, "%f",  ary+size*i+j);
		}
	}  
}
/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
void InitAry(FILE *fp, float *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		fscanf(fp, "%f",  &ary[i]);
	}
}  
/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int size, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2e ", *(ary+size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		printf("%.2e ", ary[i]);
	}
	printf("\n\n");
}
#endif

