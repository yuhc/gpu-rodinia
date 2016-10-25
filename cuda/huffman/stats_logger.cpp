/*
 * Copyright 2009 Tjark Bringewat. All rights reserved.
 */

#include "stdafx.h"
#include "stats_logger.h"
#include <cstdio>
#include <map>
#include <sstream>

std::map<std::string,unsigned int> filenames;

void LogStats(const char *graphname, const char *seriesname,
			  float xValue, float yValue,
			  const char *xAxisQuantity, const char *yAxisQuantity,
			  const char *xAxisUnit, const char *yAxisUnit,
			  const char *xAxisScaleType, const char *yAxisScaleType,
			  unsigned int seriesnumber,
			  const char *description){
	std::ostringstream temp, temp2;
	temp << graphname << "__" << seriesname;
	size_t exists = filenames.count(temp.str());
	if (!exists)
		filenames[temp.str()] = seriesnumber;
	temp2 << graphname << "__" << filenames[temp.str()] << "_" << seriesname << ".txt";
	FILE *f;
	if (!exists) {
		f = fopen(temp2.str().c_str(), "wt");
		fprintf(f, "SERIES_NAME\n%s\n", seriesname);
		fprintf(f, "X_AXIS_QUANTITY\n%s\n", xAxisQuantity);
		fprintf(f, "Y_AXIS_QUANTITY\n%s\n", yAxisQuantity);
		fprintf(f, "X_AXIS_UNIT\n%s\n", xAxisUnit);
		fprintf(f, "Y_AXIS_UNIT\n%s\n", yAxisUnit);
		fprintf(f, "X_AXIS_SCALE_TYPE\n%s\n", xAxisScaleType);
		fprintf(f, "Y_AXIS_SCALE_TYPE\n%s\n", yAxisScaleType);
		fprintf(f, "DESCRIPTION\n%s\n", description);
		fprintf(f, "__DATA__\n");
	}
	else {
		f = fopen(temp2.str().c_str(), "at");
	}
	fprintf(f, "%f %f\n", xValue, yValue);
	fclose(f);
}
