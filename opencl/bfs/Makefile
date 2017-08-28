include ../../common/make.config

#C compiler
CC = g++

SRC = bfs.cpp timer.cc

CC_FLAGS = -g -O3

EXE = bfs

release:$(SRC)
	$(CC) $(CC_FLAGS)  -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

errmsg:$(SRC)
	$(CC) $(CC_FLAGS)  -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -D ERRMSG PTX_MSG

ptx:$(SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -D PTX_MSG

profile:$(SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -D PROFILING

res:$(SRC)
	$(CC) $(CC_FLAGS) -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -D RES_MSG

debug: $(SRC)
	$(CC) $(CC_FLAGS) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL 

run:
	./$(EXE)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
