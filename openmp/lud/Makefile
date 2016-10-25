all: lud_omp lud_omp_offload

lud_omp:
	cd omp; make

lud_omp_offload:
	cd omp; make -f Makefile.offload

clean:
	cd omp; make clean ; make -f Makefile.offload clean
