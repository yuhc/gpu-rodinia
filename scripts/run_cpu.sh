#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OCLDIR=$DIR/gpu-rodinia/opencl

bm="backprop bfs cfd gaussian hotspot hotspot3D hybridsort lud \
    nn nw pathfinder srad streamcluster"

OUTDIR=$DIR/results-cpu
mkdir $OUTDIR &>/dev/null

cd $OCLDIR
exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

for b in $bm; do
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $b
    make clean ; make TYPE=CPU
    for idx in `seq 1 15`; do
        exe ./run -p 0 -d 0
        exe echo
    done
    cd $OCLDIR
    exe echo
    echo
done
