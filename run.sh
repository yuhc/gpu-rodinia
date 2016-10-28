#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OCLDIR=$DIR/opencl

# 10 in total
# dwt2d and hybridsort does not work
bm="backprop b+tree heartwall hotspot3D kmeans leukocyte myocyte \
    nw pathfinder streamcluster bfs cfd gaussian hotspot \
    lavaMD lud nn particlefilter srad"

OUTDIR=$DIR/result
mkdir $OUTDIR &>/dev/null

cd $OCLDIR
exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

for b in $bm; do
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $b
    for idx in `seq 1 3`; do
        exe sudo -E perf stat -A -a -e instructions,cache-misses,cache-references,cycles \
            ./run
        exe echo
    done
    cd $OCLDIR
    exe echo
    echo
done
