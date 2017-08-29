#include <stdio.h>
#include "timing.h"

void time_measure_start(struct timeval *tv)
{
	gettimeofday(tv, NULL);
}

void time_measure_end(struct timeval *tv)
{
	struct timeval tv_now, tv_diff;
	double d;

	gettimeofday(&tv_now, NULL);
	tvsub(&tv_now, tv, &tv_diff);

	d = (double) tv_diff.tv_sec * 1000.0 + (double) tv_diff.tv_usec / 1000.0;
	printf("Time (Memory Copy and Launch) = %f (ms)\n", d);
}
