include ../../../common/make.config

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

INC = -I$(OPENCL_INC) -I../common
LIB = -L$(OPENCL_LIB)

#C compiler
CC = g++
CC_FLAGS = -g -O3 -Wall

EXE = lud.out

$(EXE): lud.cpp
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) \
		-o $(EXE) \
		lud.cpp ../common/common.c ../../util/timing.c \
		$(INC) $(LIB) -lOpenCL \
		-I../../util -DTIMING \
		-Wno-unused-result

clean:
	rm -f ../$(EXE)
