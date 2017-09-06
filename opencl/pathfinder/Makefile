include ../../common/make.config

#Can be changed by `make TYPE=CPU`
TYPE = GPU

#Library
ifeq ($(TYPE),GPU)
	OPENCL_INC = $(NV_OPENCL_INC)
	OPENCL_LIB = $(NV_OPENCL_LIB)
else
	OPENCL_INC = $(INTEL_OPENCL_INC)
	OPENCL_LIB = $(INTEL_OPENCL_LIB)
endif

CXX = g++
DEBUGFLAGS = -g
# You will need to modify the following 2 lines
# by inserting the correct path to the OpenCL
# lib and inc directories.
CXXFLAGS = -I$(OPENCL_INC) -c $(DEBUGFLAGS)
LINKFLAGS = -L$(OPENCL_LIB)

# link objects(binaries) together
a.out: main.o OpenCL.o
	$(CXX) $(LINKFLAGS) main.o OpenCL.o timing.o \
		-o pathfinder.out \
		-lOpenCL \
	 	-I../util -DTIMING

# compile main function file into object (binary)
main.o: main.cpp OpenCL.cpp kernels.cl ../util/timing.c
	$(CXX) $(CXXFLAGS) main.cpp OpenCL.cpp ../util/timing.c -I../util -DTIMING

clean:
	clear
	@rm -f pathfinder.out *.o *.out *~ *\#

run: a.out
	./pathfinder.out -c 100000 -r 100 -h 20 -p 0 -d 0 > result.txt
