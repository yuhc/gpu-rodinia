/*
 * Copyright Tjark Bringewat. All rights reserved.
 */

#ifndef _STATS_LOGGER_H_
#define _STATS_LOGGER_H_

#include <cstring>
#pragma warning(disable:4996)

extern "C" void LogStats(const char *graphname,
						 const char *seriesname,
						 float xValue,
						 float yValue,
						 const char *xAxisQuantity,
						 const char *yAxisQuantity,
						 const char *xAxisUnit="",
						 const char *yAxisUnit="",
						 const char *xAxisScaleType="lin",
						 const char *yAxisScaleType="lin",
						 unsigned int seriesnumber=0,
						 const char *description="");

inline void LogStats2(const char *graph, // Groups several functions into one graph. Only appears in the file name.
					  const char *function, // Name of the particular function. Appears in file name and legend.
					  float yValue,
					  float xValue,
					  const char *yAxisName="Time",
					  const char *yAxisUnit="ms",
					  const char *xAxisName="Data size",
					  const char *xAxisUnit="MB",
					  const char *yAxisScaleType="lin", // Can be lin or log for linear or logarithmic scale, respectively.
					  const char *xAxisScaleType="log",
					  unsigned int fId=0, // Determines the order in which different functions are plotted to a common graph. Only appears in the file name.
					  const char *description="")
{
	LogStats(graph, function, xValue, yValue, xAxisName, yAxisName, xAxisUnit, yAxisUnit, xAxisScaleType, yAxisScaleType, fId, description);
	if (strcmp(xAxisUnit,"MB")==0 && strcmp(yAxisUnit,"ms")==0) {
		char buffer[100];
		strcpy(buffer, graph);
		strcat(buffer, "_datarate");
		LogStats(buffer, function, xValue, (xValue*1000.0f)/(yValue*1024.0f), xAxisName, "Data rate", xAxisUnit, "GB/s", xAxisScaleType, yAxisScaleType, fId, description);
	}
}

#endif
