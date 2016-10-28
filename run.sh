#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OCLDIR=$DIR/opencl

# 10 in total
bm="backprop bfs b+tree cfd heartwall hotspot hotspot3D kmeans laraMD \
leukocyte lud myocyte nw particlefilter pathfinder srad streamcluster"

OUTDIR=$DIR/result
mkdir $OUTDIR &>/dev/null

cd $OCLDIR
exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

for b in $bm; do
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $b
    exe sudo -E perf stat -A -a -e instructions,cache-misses,cache-references,cycles \
        ./run
    cd $OCLDIR
    exe echo
    echo
done
