include ../../common/make.config

# OpenCL installation path      
# OPENCL_DIR = /home/mwb7w/NVIDIA_GPU_Computing_SDK/
# OPENCL_DIR = /home/mwb7w/ati-stream-sdk-v2.2-lnx64/


CC = gcc
CC_FLAGS = -g -O3 -Wall

EXE = hotspot.out

$(EXE): hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o $(EXE) -I$(OPENCL_INC) OpenCL_helper_library.c hotspot.c -L$(OPENCL_LIB) -lOpenCL

run:
	./$(EXE) 1024 4 10000 ../../data/hotspot/temp_1024 ../../data/hotspot/power_1024 output.txt -p 0 -d 0

clean:
	rm -f $(EXE) *.txt || true
