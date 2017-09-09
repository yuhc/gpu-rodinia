#include "hotspot.h"

//Primitives for timing
#ifdef TIMING
#include "timing.h"

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
#endif

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {

	int i,j, index=0;
	FILE *fp;
	char str[STR_SIZE];

	if( (fp = fopen(file, "w" )) == 0 )
          printf( "The file was not opened\n" );


	for (i=0; i < grid_rows; i++) 
	 for (j=0; j < grid_cols; j++)
	 {

		 sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
		 fputs(str,fp);
		 index++;
	 }
		
      fclose(fp);	
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  	int i,j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if( (fp  = fopen(file, "r" )) ==0 )
            fatal( "The file was not opened" );


	for (i=0; i <= grid_rows-1; i++) 
	 for (j=0; j <= grid_cols-1; j++)
	 {
		if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
		if (feof(fp))
			fatal("not enough lines in file");
		//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
		if ((sscanf(str, "%f", &val) != 1))
			fatal("invalid file format");
		vect[i*grid_cols+j] = val;
	}

	fclose(fp);	

}


/*
   compute N time steps
*/

int compute_tran_temp(cl_mem MatrixPower, cl_mem MatrixTemp[2], int col, int row, \
		int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows,
		float *TempCPU, float *PowerCPU) 
{ 
	
	float grid_height = chip_height / row;
	float grid_width = chip_width / col;

	float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	float Rz = t_chip / (K_SI * grid_height * grid_width);

	float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	float step = PRECISION / max_slope;
	int t, itr;

	int src = 0, dst = 1;
	
	cl_int error;
	cl_event events[total_iterations/num_iterations + 1];
	
	// Determine GPU work group grid
	size_t global_work_size[2];
	global_work_size[0] = BLOCK_SIZE * blockCols;
	global_work_size[1] = BLOCK_SIZE * blockRows;
	size_t local_work_size[2];
	local_work_size[0] = BLOCK_SIZE;
	local_work_size[1] = BLOCK_SIZE;
	
	for (t = 0, itr = 0; t < total_iterations; t += num_iterations, itr++) {
		
		// Specify kernel arguments
		int iter = MIN(num_iterations, total_iterations - t);
		clSetKernelArg(kernel, 0, sizeof(int), (void *) &iter);
		clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &MatrixPower);
		clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &MatrixTemp[src]);
		clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &MatrixTemp[dst]);
		clSetKernelArg(kernel, 4, sizeof(int), (void *) &col);
		clSetKernelArg(kernel, 5, sizeof(int), (void *) &row);
		clSetKernelArg(kernel, 6, sizeof(int), (void *) &borderCols);
		clSetKernelArg(kernel, 7, sizeof(int), (void *) &borderRows);
		clSetKernelArg(kernel, 8, sizeof(float), (void *) &Cap);
		clSetKernelArg(kernel, 9, sizeof(float), (void *) &Rx);
		clSetKernelArg(kernel, 10, sizeof(float), (void *) &Ry);
		clSetKernelArg(kernel, 11, sizeof(float), (void *) &Rz);
		clSetKernelArg(kernel, 12, sizeof(float), (void *) &step);

		// Launch kernel
		error = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &events[itr]);
		if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

		// Flush the queue
		error = clFlush(command_queue);
		if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

		// Swap input and output GPU matrices
		src = 1 - src;
		dst = 1 - dst;
	}

	for (t = 0, itr = 0; t < total_iterations; t += num_iterations, itr++) {
#ifdef TIMING
	    kernel_time += probe_event_time(events[itr], command_queue);
#endif
	    clReleaseEvent(events[itr]);
	}

	return src;
}

void usage(int argc, char **argv) {
	fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file> [-p platform_id] [-d device_id]\n", argv[0]);
	fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char** argv) {

    printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

    int size;
    int grid_rows,grid_cols = 0;
    float *FilesavingTemp,*FilesavingPower; //,*MatrixOut; 
    char *tfile, *pfile, *ofile;

    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations

	if (argc < 7)
		usage(argc, argv);
	if((grid_rows = atoi(argv[1]))<=0||
	   (grid_cols = atoi(argv[1]))<=0||
       (pyramid_height = atoi(argv[2]))<=0||
       (total_iterations = atoi(argv[3]))<=0)
		usage(argc, argv);

	tfile=argv[4];
    pfile=argv[5];
    ofile=argv[6];
    size=grid_rows*grid_cols;

    // OCL config
    int platform_id_inuse = 0;            // platform id in use (default: 0)
    int device_id_inuse = 0;              //device id in use (default : 0)
    cl_device_type device_type = CL_DEVICE_TYPE_GPU;

    int cur_arg;
	for (cur_arg = 1; cur_arg<argc; cur_arg++) {
        if (strcmp(argv[cur_arg], "-h") == 0) 
		    usage(argc, argv);
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

    // --------------- pyramid parameters --------------- 
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));
    // MatrixOut = (float *) calloc (size, sizeof(float));

    if( !FilesavingPower || !FilesavingTemp) // || !MatrixOut)
        fatal("unable to allocate memory");

	// Read input data from disk
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);

#ifdef  TIMING
    gettimeofday(&tv_total_start, NULL);
#endif

	cl_int error;
	cl_uint num_platforms;
	
	// Get the number of platforms
	error = clGetPlatformIDs(0, NULL, &num_platforms);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// Get the list of platforms
	cl_platform_id* platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * num_platforms);
	error = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	// Print the chosen platform (if there are multiple platforms, choose the first one)
	cl_platform_id platform = platforms[platform_id_inuse];
	char pbuf[100];
	error = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	printf("Platform: %s\n", pbuf);

    // Get devices
	cl_uint devices_size;
	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &devices_size);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	printf("num_devices = %d\n", devices_size);
    if (device_id_inuse > devices_size) {
        printf("Invalid Device Number\n");
    	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    }
	cl_device_id *devices = (cl_device_id *)malloc(sizeof(cl_device_id)*devices_size);
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devices_size, devices, NULL);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

	device = devices[device_id_inuse];
	error = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(pbuf), pbuf, NULL);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	printf("Device: %s\n", pbuf);

	// Get device type
	error = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), (void *)&device_type, NULL);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	if (device_type == CL_DEVICE_TYPE_GPU)
        printf("Use GPU device\n");
    else if (device_type == CL_DEVICE_TYPE_CPU)
        printf("Use CPU device\n");

	// Create a GPU context
	cl_context_properties context_properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
    context = clCreateContextFromType(context_properties, device_type, NULL, NULL, &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

	// Create a command queue
#ifdef TIMING
	command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
#else
	command_queue = clCreateCommandQueue(context, device, 0, &error);
#endif
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

	// Load kernel source from file
	const char *source = load_kernel_source("hotspot_kernel.cl");
	size_t sourceSize = strlen(source);

	// Compile the kernel
    cl_program program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	
	char clOptions[110];
	//  sprintf(clOptions,"-I../../src"); 
	sprintf(clOptions," ");
#ifdef BLOCK_SIZE
	sprintf(clOptions + strlen(clOptions), " -DBLOCK_SIZE=%d", BLOCK_SIZE);
#endif

    // Create an executable from the kernel
	error = clBuildProgram(program, 1, &device, clOptions, NULL, NULL);
	// Show compiler warnings/errors
	static char log[65536]; memset(log, 0, sizeof(log));
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
	if (strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    kernel = clCreateKernel(program, "hotspot", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

	// Create two temperature matrices and copy the temperature input data
	cl_mem MatrixTemp[2];

#ifdef  TIMING
	gettimeofday(&tv_init_end, NULL);
	tvsub(&tv_init_end, &tv_total_start, &tv);
	init_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif
    // Lingjie Zhang modifited at Nov 1, 2015
    //MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * size, NULL, &error);
    MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE , sizeof(float) * size, NULL, &error);
    // end Lingjie Zhang modification
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_init_end, &tv);
    mem_alloc_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	// Create input memory buffers on device
	MatrixTemp[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingTemp, &error);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
	// Copy the power input data
	cl_mem MatrixPower = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(float) * size, FilesavingPower, &error);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

// https://software.intel.com/en-us/forums/opencl/topic/509406
// The copy should happen when the data is used; however the test shows the data is copied immediately with
//  nVidia OpenCL. w/o we include the time as h2d_time.
#ifdef  TIMING
    gettimeofday(&tv_h2d_end, NULL);
    tvsub(&tv_h2d_end, &tv_mem_alloc_end, &tv);
    h2d_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	// Perform the computation
	int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows, total_iterations, pyramid_height,
								blockCols, blockRows, borderCols, borderRows, FilesavingTemp, FilesavingPower);
	
	cl_event event;
	// Copy final temperature data back
	cl_float *MatrixOut = (cl_float *) clEnqueueMapBuffer(command_queue, MatrixTemp[ret], CL_TRUE, CL_MAP_READ, 0, sizeof(float) * size, 0, NULL, &event, &error);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
#ifdef TIMING
    d2h_time += probe_event_time(event, command_queue);
#endif
    clReleaseEvent(event);

	// Write final output to output file
    writeoutput(MatrixOut, grid_rows, grid_cols, ofile);

#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif
	error = clEnqueueUnmapMemObject(command_queue, MatrixTemp[ret], (void *) MatrixOut, 0, NULL, NULL);
	if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

	clReleaseMemObject(MatrixTemp[0]);
	clReleaseMemObject(MatrixTemp[1]);
	clReleaseMemObject(MatrixPower);

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

	return 0;
}
