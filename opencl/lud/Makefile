all: lud

lud:
	cd ocl; make
	mv ocl/lud.out .;

clean:
	cd ocl; make clean

run: lud
	#./lud.out -i ../../data/lud/256.dat
	#./lud.out -s 1024 -v
	./lud.out -s 8192 -p 0 -d 0
