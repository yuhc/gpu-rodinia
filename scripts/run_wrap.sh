#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OCLDIR=$DIR/opencl-wrap/tests

bm="backprop bfs cfd gaussian hotspot hotspot3D hybridsort lud \
    nn nw pathfinder srad streamcluster"

OUTDIR=$DIR/results-wrap
mkdir $OUTDIR &>/dev/null

cd $OCLDIR
exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

for b in $bm; do
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $b
    make clean && make TYPE=GPU
    for idx in `seq 1 10`; do
        exe ./run -p 1 -d 0
        exe echo
    done
    cd $OCLDIR
    exe echo
    echo
done
