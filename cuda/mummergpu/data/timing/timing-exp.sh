#!/bin/bash

MUMmerGPU=../../bin/linux32/release/mummergpu

export CUDA_PROFILE=1
export CUDA_PROFILE_CSV=1




trial (){
	COUNTER=0
	while [  $COUNTER -lt $1 ]; do
		echo Trial \# $(($COUNTER+1)) of $1
	    $MUMmerGPU $ARGS > $OUT 2>/dev/null
		grep Kernel cuda_profile.log  | awk -F"," -v trial=$COUNTER '{print trial,$2,$3,$4,$5,$6,$7,$8}'  >> 	$2
	    let COUNTER=COUNTER+1 
	done
}

repeat_trials (){
	COUNTER=0
	# Repeat the experiment $1 times, adding the profile output for the 
	# kernel to a growing list
	
	while [  $COUNTER -lt $1 ]; do
		# The profiler only seems to have 4 hardware counters,
		# so to collect values for all the parameters we are interested int
		# we need to run each trial several times, each with a different
		# profile configuration (the .cnf files)
		export CUDA_PROFILE_CONFIG=global.cnf
		echo Trial \# $(($COUNTER+1)) of $1
	    $MUMmerGPU $ARGS > $OUT 2>/dev/null
		grep Kernel cuda_profile.log  | awk -F"," -v trial=$COUNTER \
			'{print trial,$2,$3,$4,$5,$6,$7,$8}'  >> xxx
			
		export CUDA_PROFILE_CONFIG=instructions.cnf
		$MUMmerGPU $ARGS > $OUT 2>/dev/null
		grep Kernel cuda_profile.log  | awk -F"," -v trial=$COUNTER \
			'{print trial,$5,$6,$7,$8}'  >> yyy
	    let COUNTER=COUNTER+1 
	done
	
	# Collect the profile values for each 'trial' into a single file
	join -1 1 -2 1 xxx yyy >> $2
	rm xxx yyy
}

reset_profile_output (){
	touch $PROFILE_OUTPUT
	rm $PROFILE_OUTPUT
	echo "# trial gputime cputime occupancy gld_incoherent gld_coherent gst_incoherent gst_coherent branch divergent_branch instructions warp_serialize" > $PROFILE_OUTPUT
} 

ARGS='-l 1 divcostref.fa divcost_fulldivqry.fa'
OUT=/dev/null
PROFILE_OUTPUT=full_div.pro

reset_profile_output $PROFILE_OUTPUT
repeat_trials 5 $PROFILE_OUTPUT

ARGS='-l 1 divcostref.fa divcost_nodivqry.fa'
OUT=/dev/null
PROFILE_OUTPUT=no_div.pro

reset_profile_output $PROFILE_OUTPUT
repeat_trials 5 $PROFILE_OUTPUT