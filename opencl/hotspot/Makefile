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

CC = gcc
CC_FLAGS = -g -O3 -Wall

EXE = hotspot.out

$(EXE): hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o $(EXE) -I$(OPENCL_INC) OpenCL_helper_library.c hotspot.c ../util/timing.c -L$(OPENCL_LIB) -lOpenCL -I../util -DTIMING

run:
	./$(EXE) 1024 4 10000 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output.txt -p 0 -d 0

clean:
	rm -f $(EXE) *.txt || true
