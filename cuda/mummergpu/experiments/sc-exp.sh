#!/bin/bash

BINDIR=/fs/szdevel/cole/mummergpu/trunk/mummergpu/bin/linux32/release
DATADIR=/tmp/cole
OUTDIR=/tmp/cole/exp_out

# Run a single configuration of MUMmerGPU
run_mummergpu () {
	BIN=$1
	REF=$2
	QRY=$3
	MINMATCH=$4
	ORG=$5
	cmd="$BINDIR/$BIN -s $ORG.$BIN.$REF.$MINMATCH.$QRY.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > $OUTDIR/$ORG/$BIN.out 2>/dev/null"
	echo $cmd
	$BINDIR/$BIN -s $ORG.$BIN.$REF.$MINMATCH.$QRY.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > $OUTDIR/$ORG/$BIN.out 2>/dev/null
	
	# Checksum the output and then remove it so we don't fill the disk
	md5sum $OUTDIR/$ORG/$BIN.out > $OUTDIR/$ORG/$BIN.md5sum
	rm $OUTDIR/$ORG/$BIN.out
}

# Run all configs on a given data set, and then check that the output
# is the same for each config.
run_all_configs () {
	ORG=$1
	REF=$2
	QRY=$3
	MINMATCH=$4
	
	# Run all configs with the above params
	run_mummergpu CONTROL $REF $QRY $MINMATCH $ORG
	run_mummergpu Q $REF $QRY $MINMATCH $ORG
	run_mummergpu R $REF $QRY $MINMATCH $ORG
	run_mummergpu QR $REF $QRY $MINMATCH $ORG
	run_mummergpu T $REF $QRY $MINMATCH $ORG
	run_mummergpu QT $REF $QRY $MINMATCH $ORG
	run_mummergpu RT $REF $QRY $MINMATCH $ORG
	run_mummergpu QRT $REF $QRY $MINMATCH $ORG
	run_mummergpu m $REF $QRY $MINMATCH $ORG
	run_mummergpu Qm $REF $QRY $MINMATCH $ORG
	run_mummergpu Rm $REF $QRY $MINMATCH $ORG
	run_mummergpu QRm $REF $QRY $MINMATCH $ORG
	run_mummergpu Tm $REF $QRY $MINMATCH $ORG
	run_mummergpu QTm $REF $QRY $MINMATCH $ORG
	run_mummergpu RTm $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTm $REF $QRY $MINMATCH $ORG
	run_mummergpu r $REF $QRY $MINMATCH $ORG
	run_mummergpu Qr $REF $QRY $MINMATCH $ORG
	run_mummergpu Rr $REF $QRY $MINMATCH $ORG
	run_mummergpu QRr $REF $QRY $MINMATCH $ORG
	run_mummergpu Tr $REF $QRY $MINMATCH $ORG
	run_mummergpu QTr $REF $QRY $MINMATCH $ORG
	run_mummergpu RTr $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTr $REF $QRY $MINMATCH $ORG
	run_mummergpu mr $REF $QRY $MINMATCH $ORG
	run_mummergpu Qmr $REF $QRY $MINMATCH $ORG
	run_mummergpu Rmr $REF $QRY $MINMATCH $ORG
	run_mummergpu QRmr $REF $QRY $MINMATCH $ORG
	run_mummergpu Tmr $REF $QRY $MINMATCH $ORG
	run_mummergpu QTmr $REF $QRY $MINMATCH $ORG
	run_mummergpu RTmr $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTmr $REF $QRY $MINMATCH $ORG
	run_mummergpu t $REF $QRY $MINMATCH $ORG
	run_mummergpu Qt $REF $QRY $MINMATCH $ORG
	run_mummergpu Rt $REF $QRY $MINMATCH $ORG
	run_mummergpu QRt $REF $QRY $MINMATCH $ORG
	run_mummergpu Tt $REF $QRY $MINMATCH $ORG
	run_mummergpu QTt $REF $QRY $MINMATCH $ORG
	run_mummergpu RTt $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTt $REF $QRY $MINMATCH $ORG
	run_mummergpu mt $REF $QRY $MINMATCH $ORG
	run_mummergpu Qmt $REF $QRY $MINMATCH $ORG
	run_mummergpu Rmt $REF $QRY $MINMATCH $ORG
	run_mummergpu QRmt $REF $QRY $MINMATCH $ORG
	run_mummergpu Tmt $REF $QRY $MINMATCH $ORG
	run_mummergpu QTmt $REF $QRY $MINMATCH $ORG
	run_mummergpu RTmt $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTmt $REF $QRY $MINMATCH $ORG
	run_mummergpu rt $REF $QRY $MINMATCH $ORG
	run_mummergpu Qrt $REF $QRY $MINMATCH $ORG
	run_mummergpu Rrt $REF $QRY $MINMATCH $ORG
	run_mummergpu QRrt $REF $QRY $MINMATCH $ORG
	run_mummergpu Trt $REF $QRY $MINMATCH $ORG
	run_mummergpu QTrt $REF $QRY $MINMATCH $ORG
	run_mummergpu RTrt $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTrt $REF $QRY $MINMATCH $ORG
	run_mummergpu mrt $REF $QRY $MINMATCH $ORG
	run_mummergpu Qmrt $REF $QRY $MINMATCH $ORG
	run_mummergpu Rmrt $REF $QRY $MINMATCH $ORG
	run_mummergpu QRmrt $REF $QRY $MINMATCH $ORG
	run_mummergpu Tmrt $REF $QRY $MINMATCH $ORG
	run_mummergpu QTmrt $REF $QRY $MINMATCH $ORG
	run_mummergpu RTmrt $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTmrt $REF $QRY $MINMATCH $ORG
	run_mummergpu n $REF $QRY $MINMATCH $ORG
	run_mummergpu Qn $REF $QRY $MINMATCH $ORG
	run_mummergpu Rn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRn $REF $QRY $MINMATCH $ORG
	run_mummergpu Tn $REF $QRY $MINMATCH $ORG
	run_mummergpu QTn $REF $QRY $MINMATCH $ORG
	run_mummergpu RTn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTn $REF $QRY $MINMATCH $ORG
	run_mummergpu mn $REF $QRY $MINMATCH $ORG
	run_mummergpu Qmn $REF $QRY $MINMATCH $ORG
	run_mummergpu Rmn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRmn $REF $QRY $MINMATCH $ORG
	run_mummergpu Tmn $REF $QRY $MINMATCH $ORG
	run_mummergpu QTmn $REF $QRY $MINMATCH $ORG
	run_mummergpu RTmn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTmn $REF $QRY $MINMATCH $ORG
	run_mummergpu rn $REF $QRY $MINMATCH $ORG
	run_mummergpu Qrn $REF $QRY $MINMATCH $ORG
	run_mummergpu Rrn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRrn $REF $QRY $MINMATCH $ORG
	run_mummergpu Trn $REF $QRY $MINMATCH $ORG
	run_mummergpu QTrn $REF $QRY $MINMATCH $ORG
	run_mummergpu RTrn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTrn $REF $QRY $MINMATCH $ORG
	run_mummergpu mrn $REF $QRY $MINMATCH $ORG
	run_mummergpu Qmrn $REF $QRY $MINMATCH $ORG
	run_mummergpu Rmrn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRmrn $REF $QRY $MINMATCH $ORG
	run_mummergpu Tmrn $REF $QRY $MINMATCH $ORG
	run_mummergpu QTmrn $REF $QRY $MINMATCH $ORG
	run_mummergpu RTmrn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTmrn $REF $QRY $MINMATCH $ORG
	run_mummergpu tn $REF $QRY $MINMATCH $ORG
	run_mummergpu Qtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Rtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Ttn $REF $QRY $MINMATCH $ORG
	run_mummergpu QTtn $REF $QRY $MINMATCH $ORG
	run_mummergpu RTtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTtn $REF $QRY $MINMATCH $ORG
	run_mummergpu mtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Qmtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Rmtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRmtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Tmtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QTmtn $REF $QRY $MINMATCH $ORG
	run_mummergpu RTmtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTmtn $REF $QRY $MINMATCH $ORG
	run_mummergpu rtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Qrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Rrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Trtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QTrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu RTrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu mrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Qmrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Rmrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRmrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu Tmrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QTmrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu RTmrtn $REF $QRY $MINMATCH $ORG
	run_mummergpu QRTmrtn $REF $QRY $MINMATCH $ORG

	
	# # Collect the individual md5sums in one place
	cat $OUTDIR/$ORG/*.md5sum > $ORG.$REF.$QRY.$MINMATCH.md5
	rm $OUTDIR/$ORG/*.md5sum
	
	STATS=$ORG.$REF.$QRY.$MINMATCH.allstats
	rm $STATS
	touch $STATS
	echo -n "Configuration," >> $STATS
	head -n1 -q $ORG.*.$REF.$MINMATCH.$QRY.stats | uniq >> $STATS
	for i in $( ls $ORG.*.$REF.$MINMATCH.$QRY.stats ); do
		BIN=`echo $i | sed "s/.$REF.$MINMATCH.$QRY.stats//" | sed "s/$ORG.//"`
		echo $i
		echo $BIN
		echo -n "$BIN," >> $STATS 
		awk 'NR != 1 {print}' < $i >> $STATS
		rm $i
	done
	
}

run_increasing_length_config()
{
	ORG=anthrax
	REF=NC_003997.fna
	CONFIG=$1
	
	run_mummergpu $CONFIG $REF NC_003997_q25bp.fna 25 $ORG
	run_mummergpu $CONFIG $REF NC_003997_q50bp.fna 50 $ORG
	run_mummergpu $CONFIG $REF NC_003997_q100bp.fna 100 $ORG
	run_mummergpu $CONFIG $REF NC_003997_q200bp.fna 200 $ORG
	run_mummergpu $CONFIG $REF NC_003997_q400bp.fna 400 $ORG
	run_mummergpu $CONFIG $REF NC_003997_q800bp.fna 800 $ORG
}

# Run a single configuration of MUMmerGPU
run_mummergpu_C () {
	BIN=$1
	REF=$2
	QRY=$3
	MINMATCH=$4
	ORG=$5
	cmd="$BINDIR/$BIN -C -s $ORG.$BIN.$QRY.C.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > $OUTDIR/$ORG/$BIN.out 2>/dev/null"
	echo $cmd
	$BINDIR/$BIN -C -s $ORG.$BIN.$QRY.C.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > $OUTDIR/$ORG/$BIN.out 2>/dev/null
	
	# Checksum the output and then remove it so we don't fill the disk
	md5sum $OUTDIR/$ORG/$BIN.out > $OUTDIR/$ORG/$BIN.md5sum
	rm $OUTDIR/$ORG/$BIN.out
}


# Run a single configuration of MUMmerGPU
run_mummergpu_no_out () {
	BIN=$1
	REF=$2
	QRY=$3
	MINMATCH=$4
	ORG=$5
	cmd="$BINDIR/$BIN -s $ORG.$BIN.$QRY.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > /dev/null 2>/dev/null"
	echo $cmd
	$BINDIR/$BIN -s $ORG.$BIN.$QRY.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > /dev/null 2>/dev/null
}

# Run a single configuration of MUMmerGPU on the CPU
run_mummergpu_C_no_out () {
	BIN=$1
	REF=$2
	QRY=$3
	MINMATCH=$4
	ORG=$5
	cmd="$BINDIR/$BIN -C -s $ORG.$BIN.$QRY.C.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > /dev/null 2>/dev/null"
	echo $cmd
	$BINDIR/$BIN -C -s $ORG.$BIN.$QRY.C.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > /dev/null 2>/dev/null
}

run_cpu_configs()
{
	ORG=$1
	REF=$2
	QRY=$3
	MINMATCH=$4

	# Run all configs with the above params
	run_mummergpu_C CONTROL $REF $QRY $MINMATCH $ORG
	run_mummergpu_C m $REF $QRY $MINMATCH $ORG
	run_mummergpu_C t $REF $QRY $MINMATCH $ORG
	run_mummergpu_C n $REF $QRY $MINMATCH $ORG
	run_mummergpu_C mt $REF $QRY $MINMATCH $ORG
	run_mummergpu_C tn $REF $QRY $MINMATCH $ORG
	run_mummergpu_C mn $REF $QRY $MINMATCH $ORG
	run_mummergpu_C mtn $REF $QRY $MINMATCH $ORG


	# # Collect the individual md5sums in one place
	cat $OUTDIR/$ORG/*.md5sum > $ORG.$REF.$QRY.$MINMATCH.md5
	rm $OUTDIR/$ORG/*.md5sum

	STATS=$ORG.$REF.$QRY.$MINMATCH.C.allstats
	rm $STATS
	touch $STATS
	echo -n "Configuration," >> $STATS
	head -n1 -q *.stats | uniq >> $STATS
	for i in $( ls $ORG.*.stats ); do
		BIN=`echo $i | sed 's/.stats//' | sed "s/$ORG.//"`
		echo -n "$BIN," >> $STATS 
		awk 'NR != 1 {print}' < $i >> $STATS
		rm $i
	done
}

run_increasing_length()
{
	ORG=anthrax
	REF=NC_003997.fna
	
	run_increasing_length_config R

	# Collect the individual md5sums in one place
	cat $OUTDIR/$ORG/*.md5sum > $ORG.$REF.$QRY.$MINMATCH.md5
	rm $OUTDIR/$ORG/*.md5sum

	STATS=anthrax_increasing_length.allstats
	rm $STATS
	touch $STATS
	echo -n "Configuration," >> $STATS
	head -n1 -q *.stats | uniq >> $STATS
	for i in $( ls $ORG.*.stats ); do
		BIN=`echo $i | sed 's/.stats//' | sed "s/$ORG.//"`
		echo -n "$BIN," >> $STATS 
		awk 'NR != 1 {print}' < $i >> $STATS
		rm $i
	done

}

run_mummergpu_C_1()
{
	BIN=/nfshomes/cole/bin/mummergpu-1.0
	REF=$1
	QRY=$2
	MINMATCH=$3
	ORG=$4
	cmd="$BIN -C -s $ORG.mummergpu-1.0.$QRY.C.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > /dev/null 2>/dev/null"
	echo $cmd
	$BIN -C -s $ORG.mummergpu-1.0.$QRY.C.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > /dev/null 2>/dev/null
}

run_mummergpu_1()
{
	BIN=/nfshomes/cole/bin/mummergpu-1.0
	REF=$1
	QRY=$2
	MINMATCH=$3
	ORG=$4
	cmd="$BIN -s $ORG.mummergpu-1.0.$QRY.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > /dev/null 2>/dev/null"
	echo $cmd
	$BIN -s $ORG.mummergpu-1.0.$QRY.stats -l $MINMATCH $DATADIR/$ORG/$REF $DATADIR/$ORG/$QRY > /dev/null 2>/dev/null
}

run_end_to_end()
{
	mkdir -p e2e
	cd e2e
	
	run_mummergpu_no_out   Tn two_pages.fna half_million_reads.fna 14 h_sapiens
	run_mummergpu_C_no_out m  two_pages.fna half_million_reads.fna 14 h_sapiens
	# 
	#run_mummergpu_no_out Tn   cleanref.fna million_reads.fna 20 lmonocytogenes
	# run_mummergpu_C_no_out m cleanref.fna million_reads.fna 20 lmonocytogenes
	# 
	#run_mummergpu_no_out Tn   cleanref.fna half_million_reads.fna 100 cbriggsae
	# run_mummergpu_C_no_out m cleanref.fna half_million_reads.fna 100 cbriggsae
	# 
	#run_mummergpu Tn   cleanref.fna million_reads.fna 10 s_suis
	# run_mummergpu_C_no_out m cleanref.fna million_reads.fna 10 s_suis
	
	run_mummergpu_1   two_pages.fna half_million_reads.fna 14 h_sapiens
	run_mummergpu_C_1 two_pages.fna half_million_reads.fna 14 h_sapiens
	
	#run_mummergpu_1 cleanref.fna million_reads.fna 20 lmonocytogenes
	#run_mummergpu_C_1 cleanref.fna million_reads.fna 20 lmonocytogenes
	
	# run_mummergpu_1 cleanref.fna half_million_reads.fna 100 cbriggsae
	# run_mummergpu_C_1 cleanref.fna half_million_reads.fna 100 cbriggsae
	
	#run_mummergpu_1 cleanref.fna million_reads.fna 10 s_suis
	#run_mummergpu_C_1 cleanref.fna million_reads.fna 10 s_suis
}

run_end_to_end

#run_all_configs s_suis cleanref.fna million_reads.fna 10

#run_all_configs h_sapiens two_pages.fna half_million_reads.fna 14

#run_all_configs lmonocytogenes cleanref.fna million_reads.fna 20

#run_all_configs cbriggsae cleanref.fna half_million_reads.fna 100

#run_increasing_length

#run_cpu_configs h_sapiens two_pages.fna million_reads.fna 15

#run_cpu_configs s_suis cleanref.fna million_reads.fna 10

#run_cpu_configs lmonocytogenes cleanref.fna million_reads.fna 20

#run_cpu_configs cbriggsae cleanref.fna half_million_reads.fna 100
