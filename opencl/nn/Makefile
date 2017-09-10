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

SRC = nearestNeighbor.cpp clutils.cpp utils.cpp

CC_FLAGS = -g -O3

EXE = nn.out

release:
	$(CC) $(CC_FLAGS)  -fopenmp $(SRC) -o $(EXE) \
		-I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL \
		-I../util -DTIMING \
		-Wno-unused-result

clean:
	rm -f $(EXE)

run: release
	./$(EXE) filelist.txt -p 0 -d 0 -r 50 -lat 30 -lng 90 -q -t ~/gpu-rodinia/data -p 0 -d 0
