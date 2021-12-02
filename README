Rodinia Benchmark Suite 3.1
===========================

0. Modifications
Please download the dataset from the [original package][data], or my personal [Dropbox][dropbox] if the Virginia's link is down.
[data]: http://lava.cs.virginia.edu/Rodinia/download_links.htm
[dropbox]: https://www.dropbox.com/s/cc6cozpboht3mtu/rodinia-3.1-data.tar.gz

The Dropbox version has
MD5: 0CF70BC7BD538794F626C0E04A6447AE
SHA-256: B90994D5208EC5A0A133DFB9AB7928A1E8A16741503A91D212884B9E4FCE8CD8
CRC32: 303E69CE

Make sure the Platform 0 is the one needed to be tested, and GPU is the
device 0 on this platform.

I. Overview

The University of Virginia Rodinia Benchmark Suite is a collection of parallel programs which targets 
heterogeneous computing platforms with both multicore CPUs and GPUs.

II. Usage

1. Pakage Structure

rodinia_2.1/bin		: binary executables
rodinia_2.1/common	: common configuration file
rodinia_2.1/cuda	: source code for the CUDA implementations
rodinia_2.1/data	: input files
rodinia_2.1/openmp	: source code for the OpenMP implementations
rodinia_2.1/opencl	: source code for the OpenCL implementations

2. Build Rodinia

Install the CUDA/OCL drivers, SDK and toolkit on your machine. 

Modify the rodinia_2.1/common/make.config file to change the settings of rodinia home directory and CUDA/OCL library paths.

To compile all the programs of the Rodinia benchmark suite, simply use the universal make file to compile all the programs, or go to each 
benchmark directory and make individual programs. 

3. Run Rodinia

There is a 'run' file specifying the sample command to run each program.

IV. Change Log
Dec. 12, 2015: Rodinia 3.1 is released
********************************************************
1. Bug fix
1). OpenCL version Hotspot (Thanks Shuai Che from AMD)
    Delete this parameter "CL_MEM_ALLOC_HOST_PTR" for device-side buffer allocation.
2).  OpenCL version Kmeans (Thanks Jeroen Ketema from Imperial College London, Tzu-Te from National Chiao Tung University, Shuai Che and Michael Boyer form AMD )
    Fix data race problem for reduce kernel.
3).  OpenCL version Leukocyte (Thanks Jeroen Ketema from Imperial College London)
    Fix data race problem for find_ellipse kernel.
4).  OpenCL version srad (Thanks Jeroen Ketema from Imperial College London)
    Fix data race problem for reduce kernel
5).  OpenCL version dwt2d (Thanks Tzu-Te from National Chiao Tung University)
    Fix a bug for buffer size. 

2. New benchmarks (Thanks Linh Nguyen from Hampden-Sydney College)
1).  Hotspot3D(CUDA, OpenMP and OpenCL version)
2).  Huffman (only CUDA version)

3. Performance improvement
1). Openmp version nn (Thanks Shuai Che from AMD)
2). OpenCL version nw (Thanks Shuai Che from AMD)
3). CUDA version cfd (Thanks Ke)

5. Several OpenMP benchmarks have been improved (Thanks Sergey Vinogradov and Julia Fedorova from Intel)
1). BFS
2). LUD
3). HotSpot
4). CFD
5). NW



Mar. 02, 2013: Rodinia 2.3 is released
***********************************************************************
A.   General
Add -lOpenCL in the OPENCL_LIB definition in common/make.config
OPENCL_LIB = $(OPENCL_DIR)/OpenCL/common/lib -lOpenCL (gcc-4.6+ compatible) 

B.  OpenCL
1. Particlefilter OpenCL
a) Runtime work group size selection based on device limits
b) Several bugs of kernel fixed
c) Initialize all arrays on host side and device side
d) Fix objxy_GPU array across boundary access on device
     objxy_GPU = clCreateBuffer(context, CL_MEM_READ_WRITE, 2*sizeof (int) *countOnes, NULL, &err);
      and 
    err = clEnqueueWriteBuffer(cmd_queue, objxy_GPU, 1, 0, 2*sizeof (int) *countOnes, objxy, 0, 0, 0);
e) #define PI  3.1415926535897932  in ex_particle_OCL_naive_seq.cpp
f) put  -lOpenCL just behind -L$(OPENCL_LIB) in Makefile. 
g) delete an useless function tex1Dfetch() from particle_float.cl.
h) add single precision version!

2. B+Tree OpenCL
a) Replace CUDA function __syncthreads() with OpenCL barrier(CLK_LOCAL_MEM_FENCE) in kernel file


3. Heartwall OpenCL
a) Lower work item size from 512 to 256 (Better compatibility with AMD GPU)
b) Several bugs fixed on kernel codes
c) Several bugs fixed on host codes

4. BSF OpenCL  
a). Replace all bool with char since bool is NOT a valid type for OpenCL arguments .
b). -lOpenCL just behind -L$(OPENCL_LIB) in Makefile. (gcc-4.6+ compatible) 
c). remove NVIDIA-specific parameters and decrease thread block size for Better compatibility with AMD GPU
BFS/CLHelper.h: 
//std::string options= "-cl-nv-verbose"; // doesn't work on AMD machines
resultCL = clBuildProgram(oclHandles.program, deviceListSize, oclHandles.devices, NULL, NULL,? NULL);

bfs.cpp:
#define MAX_THREADS_PER_BLOCK 256 // 512 is too big for my AMD Fusion GPU

d) Correct bad mallocs
BFS/CLHelper.h
oclHandles.devices = (cl_device_id *)malloc(deviceListSize * sizeof(cl_device_id));

d_mem = clCreateBuffer(oclHandles.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, h_mem_ptr, &oclHandles.cl_status);

d_mem = clCreateBuffer(oclHandles.context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, size, h_mem_ptr, &oclHandles.cl_status);

h_mem_pinned = (cl_float *)clEnqueueMapBuffer(oclHandles.queue, d_mem_pinned, CL_TRUE,? \
 CL_MAP_WRITE, 0, size, 0, NULL,? \

bfs.cpp
d_graph_mask = _clMallocRW(no_of_nodes*sizeof(bool), h_graph_mask);
d_updating_graph_mask = _clMallocRW(no_of_nodes*sizeof(bool), h_updating_graph_mask);
d_graph_visited = _clMallocRW(no_of_nodes*sizeof(bool), h_graph_visited);

compare_results<int>(h_cost_ref, h_cost, no_of_nodes);

f)  Add #include <cstdlib> in bfs.cpp
g) Conditional including time.h 

5. CFD OpenCL
a) Comment out two useless clWaitForEvents commands in CLHelper.h. It will get 1.5X speedup on some GPUs. 
b) -lOpenCL just behind -L$(OPENCL_LIB) in Makefile. (gcc-4.6+ compatible) 
c) cfd/CLHelper.h
oclHandles.devices = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceListSize);

6. Backprop OpenCL. 
a) Opencl doesn’t support integer log2 and pow
backprop_kernel.cl 40 & 42 To:
for ( int i = 1 ; i <= HEIGHT ; i=i*2){                                      
  int power_two = i;
b) Change if( device_list ) delete device_list; to
if( device_list ) delete[] device_list; 

7. gaussianElim OpenCL
a) Add codes to release device buffer at the end of ForwardSub() function (gaussianElim.cpp)
b) gaussian/gaussianElim.cpp 
Add cl_cleanup();   after free(finalVec);
8. Lavamd OpenCL: In lavaMD/kernel/kernel_gpu_opencl_wrapper.c
add : #include <string.h>

9. pathfinder OpenCL
a) OpenCL.cpp: add #include <cstdlib>
b) Makefile: Changed the plase of -lOpenCL for better compatibility of gcc-4.6+.
10. streamcluster OpenCL: In CLHelper.h
oclHandles.devices = (cl_device_id *)malloc(sizeof(cl_device_id)*deviceListSize);
11. Hotspot OpenCL: In hotspot.c add clReleaseContext(context); 
before main function return.
12. kmeans OpenCL: Add shutdown() in main function to release CL resource before quit.

C. CUDA
1. CFD CUDA: solve compatablity problem with CUDA 5.0.
2. Backprop CUDA: Correct include command in backprop_cuda.cu
3. BFS CUDA: Correct include command in backprop_cuda.cu
4. kmeans CUDA: Add “-lm” in link command.
5. nn CUDA: Fix makefile bugs
6. mummergpu CUDA 
a) add #include <stdint.h>  to
mummergpu_gold.cpp
mummergpu_main.cpp
suffix-tree.cpp
b) mummergpu.cu:  correct void boardMemory function parameters types.
c) Rename getRef function to getRefGold in mummergpu_gold.cpp to avoid multiple definition

D. OpenMP
1. Kmeans OpenMP
Rename variable max_dist to min_dist in kmeans_clustering.c in kmeans_openmp/ and kmeans_serial/ folders to avoid misunderstanding. 
***********************************************************************
For bug reports and fixes:
Thanks Alexey Kravets, Georgia Kouveli and Elena Stohr from CARP project. Thanks Maxim Perminov from Intel.Thanks Daniel Lustig from Princeton. Thanks John Andrew Stratton from UIUC. Thanks Mona Jalal from University of Wisconsin.


Oct. 09, 2012: Rodinia 2.2 is released
        - BFS: Delete invalid flag CL_MEM_USE_HOST_PTR from _clMallocRW and _clMalloc functions in opencl verion. Thanks Alexey Kravets (CARP European research project).
        - Hotspot: hotspot_kernel.cl:61 correct the index calculation as grid_cols *loadYidx + loadXidx. Correct the same problem in hotspot.cu:152. Thanks Alexey Kravets.
        - Pathfinder: Added two __syncthreads in dynproc_kernel function of CUDA version to avoid data race. Thanks Ronny Krashinsky(Nvidia company) and Jiayuan Meng(Argonne National Laboratory). Alexey Kravets found and corrected the same problem in opencl version. 
        - SRAD: Replace CUDA function __syncthreads() in srad OpenCL kernel with OpenCL barrier(CLK_LOCAL_MEM_FENCE).
        - NN: Fixed the bug of CUDA version on certain input sizes. The new version detects excess of x-dimension size limit of a CUDA block grid and executes a two-dimensional grid if needed.(Only cuda version has this problem)
        - Promote B+Tree to main distribution (with output)
        - Promote Myocyte to main distribution (with output)
	
June 27, 2012: Rodinia 2.1 is released
	- Include fixes for SRAD, Heartwall, Particle Filter and Streamcluster
Nov 23, 2011: Rodinia 2.0.1 is released
	- Include a CUDA version of NN comparable to the OCL version.
	- Use a new version of clutils that is BSD, not GPL.
Nov 11, 2011: Rodinia 2.0 is released
	- Include several applications into the main suite:
	  lavaMD, Gaussian Elimination, Pathfinder, k-Nearest Neighbor and Particle Filter. 
	  Detailed application information can also be found at http://lava.cs.virginia.edu/wiki/rodinia
	- Merge new OpenCL implementations into the main tarball.
Mar 01, 2010: Rodinia 1.0 is released

III. Contact
Ke Wang: kw5na@virginia.edu
Shuai Che: sc5nf@cs.virginia.edu
Kevin Skadron: skadron@cs.virginia.edu

Rodinia wiki: 

http://lava.cs.virginia.edu/wiki/rodinia
