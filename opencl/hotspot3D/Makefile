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

ifndef CC
	CC = gcc
endif

OUTPUT = *.txt
CCFLAGS=-O3 -lm 

LIBS = -lOpenCL 

hotspot3D.out: 3D.c CL_helper.c ../util/timing.c
	$(CC) $^ $(CCFLAGS) -I$(OPENCL_INC) -L$(OPENCL_LIB) $(LIBS) \
		-I../util -DTIMING \
		-o $@

clean:
	rm -f hotspot3D.out $(OUTPUT) 
