#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <CL/cl.h>
#include "CL_helper.h"
#include "timing.h"

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

/* required precision in degrees	*/
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

#define WG_SIZE_X (64)
#define WG_SIZE_Y (4)
float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <-n rows/cols> <-l layers> <-i iterations> <-f powerFile tempFile outputFile> [-p platform] [-d device]\n", argv[0]);
  fprintf(stderr, "\t<rows/cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>  - number of layers in the grid (positive integer)\n");

  fprintf(stderr, "\t<iteration> - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile - output file\n");
  exit(1);
}


int main(int argc, char** argv)
{
    char *pfile = NULL, *tfile = NULL, *ofile = NULL;
    int iterations = 0;
    int numCols    = 0;
    int numRows    = 0;
    int layers     = 0;

    int platform_id_inuse = 0;            // platform id in use (default: 0)
    int device_id_inuse = 0;              // device id in use (default : 0)

    int cur_arg;
	for (cur_arg = 1; cur_arg<argc; cur_arg++) {
        if (strcmp(argv[cur_arg], "-h") == 0) {
			usage(argc, argv);
        }
        else if (strcmp(argv[cur_arg], "-n") == 0) {
            if (argc >= cur_arg + 1) {
                numCols = numRows = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-l") == 0) {
            if (argc >= cur_arg + 1) {
                layers = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-i") == 0) {
            if (argc >= cur_arg + 1) {
                iterations = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-f") == 0) {
            if (argc >= cur_arg + 3) {
                pfile = argv[cur_arg+1];
                tfile = argv[cur_arg+2];
                ofile = argv[cur_arg+3];
                cur_arg += 3;
            }
        }
        else if (strcmp(argv[cur_arg], "-p") == 0) {
            if (argc >= cur_arg + 1) {
                platform_id_inuse = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
        else if (strcmp(argv[cur_arg], "-d") == 0) {
            if (argc >= cur_arg + 1) {
                device_id_inuse = atoi(argv[cur_arg+1]);
                cur_arg++;
            }
        }
    }

	if (numCols == 0 || layers == 0 || iterations == 0 || pfile == NULL) {
		usage(argc, argv);
	}

  /* calculating parameters*/

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw                                              = stepDivCap/ Rx;
  cn               = cs                                              = stepDivCap/ Ry;
  ct               = cb                                              = stepDivCap/ Rz;

  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);


  int          err;           
  int size = numCols * numRows * layers;
  float*        tIn      = (float*) calloc(size,sizeof(float));
  float*        pIn      = (float*) calloc(size,sizeof(float));
  float*        tempCopy = (float*)malloc(size * sizeof(float));
  float*        tempOut  = (float*) calloc(size,sizeof(float));
  int i                 = 0;
  int count = size;
  readinput(tIn,numRows, numCols, layers, tfile);
  readinput(pIn,numRows, numCols, layers, pfile);

  size_t global[2];                   
  size_t local[2];
  memcpy(tempCopy,tIn, size * sizeof(float));

  cl_device_id     device_id;     
  cl_context       context;       
  cl_command_queue commands;      
  cl_program       program;       
  cl_kernel        ko_vadd;       

  cl_mem d_a;                     
  cl_mem d_b;                     
  cl_mem d_c;                     
  const char *KernelSource = load_kernel_source("hotspotKernel.cl"); 
  cl_uint numPlatforms;

#ifdef TIMING
  struct timeval tv;
  struct timeval tv_total_start, tv_total_end;
  struct timeval tv_init_end;
  struct timeval tv_h2d_start, tv_h2d_end;
  struct timeval tv_d2h_start, tv_d2h_end;
  struct timeval tv_kernel_start, tv_kernel_end;
  struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
  struct timeval tv_close_start, tv_close_end;
  float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
              d2h_time = 0, close_time = 0, total_time = 0;

  gettimeofday(&tv_total_start, NULL);
#endif

  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms <= 0)
    {
      printf("Error: Failed to find a platform!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  cl_platform_id Platform[numPlatforms];
  err = clGetPlatformIDs(numPlatforms, Platform, NULL);
  if (err != CL_SUCCESS || numPlatforms <= 0)
    {
      printf("Error: Failed to get the platform!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

    // get device
    cl_uint num_devices;
    err = clGetDeviceIDs(Platform[platform_id_inuse], CL_DEVICE_TYPE_ALL, 0,
            NULL, &num_devices);
    if (err != CL_SUCCESS) {
        printf("ERROR: clGetDeviceIDs failed\n");
        return EXIT_FAILURE;
    };
	printf("num_devices = %d\n", num_devices);
    if (device_id_inuse > num_devices) {
        printf("Invalid Device Number\n");
        return EXIT_FAILURE;
    }
	cl_device_id *device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
	if ( !device_list ) {
        printf("ERROR: new cl_device_id[] failed\n");
        return EXIT_FAILURE;
    }
    err = clGetDeviceIDs(Platform[platform_id_inuse], CL_DEVICE_TYPE_ALL, num_devices,
            device_list, NULL);
    if (err != CL_SUCCESS) {
        printf("ERROR: clGetDeviceIDs failed\n");
        return EXIT_FAILURE;
    };
    device_id = device_list[device_id_inuse];
    free(device_list);
  err = output_device_info(device_id);

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
    {
      printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

#ifdef TIMING
  commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
#else
  commands = clCreateCommandQueue(context, device_id, 0, &err);
#endif
  if (!commands)
    {
      printf("Error: Failed to create a command commands!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
  if (!program)
    {
      printf("Error: Failed to create compute program!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t len;
      char buffer[2048];

      printf("Error: Failed to build program executable!\n%s\n", err_code(err));
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      return EXIT_FAILURE;
    }

  ko_vadd = clCreateKernel(program, "hotspotOpt1", &err);
  if (!ko_vadd || err != CL_SUCCESS)
    {
      printf("Error: Failed to create compute kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }
#ifdef  TIMING
  gettimeofday(&tv_init_end, NULL);
  tvsub(&tv_init_end, &tv_total_start, &tv);
  init_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

  d_a  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);
  d_b  = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(float) * count, NULL, NULL);
  d_c  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);
#ifdef  TIMING
  gettimeofday(&tv_mem_alloc_end, NULL);
  tvsub(&tv_mem_alloc_end, &tv_init_end, &tv);
  mem_alloc_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

  if (!d_a || !d_b || !d_c) 
    {
      printf("Error: Failed to allocate device memory!\n");
      exit(1);
    }    

  cl_event event;
  err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, tIn, 0, NULL, &event);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write tIn to source array!\n%s\n", err_code(err));
      exit(1);
    }
#ifdef TIMING
  h2d_time += probe_event_time(event, commands);
#endif
  clReleaseEvent(event);

  err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, pIn, 0, NULL, &event);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write pIn to source array!\n%s\n", err_code(err));
      exit(1);
    }
#ifdef TIMING
    h2d_time += probe_event_time(event, commands);
#endif
    clReleaseEvent(event);

  err = clEnqueueWriteBuffer(commands, d_c, CL_TRUE, 0, sizeof(float) * count, tempOut, 0, NULL, &event);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write tempOut to source array!\n%s\n", err_code(err));
      exit(1);
    }
#ifdef TIMING
    h2d_time += probe_event_time(event, commands);
#endif
    clReleaseEvent(event);

  int j;
  for(j = 0; j < iterations; j++)
    {
      err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_b);
      err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_a);
      err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
      err |= clSetKernelArg(ko_vadd, 3, sizeof(float), &stepDivCap);
      err |= clSetKernelArg(ko_vadd, 4, sizeof(int), &numCols);
      err |= clSetKernelArg(ko_vadd, 5, sizeof(int), &numRows);
      err |= clSetKernelArg(ko_vadd, 6, sizeof(int), &layers);
      err |= clSetKernelArg(ko_vadd, 7, sizeof(float), &ce);
      err |= clSetKernelArg(ko_vadd, 8, sizeof(float), &cw);
      err |= clSetKernelArg(ko_vadd, 9, sizeof(float), &cn);
      err |= clSetKernelArg(ko_vadd, 10, sizeof(float), &cs);
      err |= clSetKernelArg(ko_vadd, 11, sizeof(float), &ct);
      err |= clSetKernelArg(ko_vadd, 12, sizeof(float), &cb);      
      err |= clSetKernelArg(ko_vadd, 13, sizeof(float), &cc);
      if (err != CL_SUCCESS)
        {
          printf("Error: Failed to set kernel arguments!\n");
          exit(1);
        }

      global[0] = numCols;
      global[1] = numRows;

      local[0] = WG_SIZE_X;
      local[1] = WG_SIZE_Y;

      err = clEnqueueNDRangeKernel(commands, ko_vadd, 2, NULL, global, local, 0, NULL, &event);
      if (err)
        {
          printf("Error: Failed to execute kernel!\n%s\n", err_code(err));
          return EXIT_FAILURE;
        }

      cl_mem temp = d_a;
      d_a         = d_c;
      d_c         = temp;

#ifdef TIMING
      kernel_time += probe_event_time(event, commands);
#endif
      clReleaseEvent(event);
    }

  clFinish(commands);
  err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(float) * count, tempOut, 0, NULL, &event);  
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to read output array!\n%s\n", err_code(err));
      exit(1);
    }
#ifdef TIMING
  d2h_time += probe_event_time(event, commands);
#endif
  clReleaseEvent(event);

#ifdef  TIMING
  gettimeofday(&tv_close_start, NULL);
#endif
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(ko_vadd);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
#ifdef  TIMING
  gettimeofday(&tv_close_end, NULL);
  tvsub(&tv_close_end, &tv_close_start, &tv);
  close_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
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

  float* answer = (float*)calloc(size, sizeof(float));
  computeTempCPU(pIn, tempCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);
  float acc = accuracy(tempOut,answer,numRows*numCols*layers);
  printf("Accuracy: %e\n",acc);

  writeoutput(tempOut,numRows,numCols,layers,ofile);

  return 0;
}

