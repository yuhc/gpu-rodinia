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

#C compiler
CC = g++

SRC = clutils.cpp gaussianElim.cpp utils.cpp

CC_FLAGS = -g -O3

EXE = gaussian.out

release:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -I../util -DTIMING -Wno-unused-result

run:
	#./gaussian.out -p 0 -d 0 -f ../../data/gaussian/matrix1024.txt
	./gaussian.out -s 2048 -p 0 -d 0

clean: 
	rm -f $(EXE) 
