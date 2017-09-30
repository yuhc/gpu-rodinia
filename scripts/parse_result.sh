#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bm="backprop bfs cfd gaussian hotspot hotspot3D hybridsort lud \
    nn nw pathfinder srad streamcluster"
col="Init MemAlloc HtoD Exec DtoH Close Total"

OUTDIR=$DIR/results-gpu
TMPFILE=$DIR/.tmp.tmp

cd $OUTDIR

echo -n "GPU,"; echo $col | tr ' ' ','
for b in $bm; do
    echo -n $b
    result=""
    echo -n > $TMPFILE
    for c in $col; do
        grep $c $OUTDIR/$b.txt | \
            awk '{ total += $2; count++ } END { print total/count }' \
            >> $TMPFILE
    done
    echo -n ","; cat $TMPFILE | paste -sd "," -
    rm $TMPFILE
done
