#!/usr/bin/env python
# encoding: utf-8
"""
untitled.py

Created by Cole Trapnell on 2008-03-22.
Copyright (c) 2008 Cole Trapnell. All rights reserved.
"""

import sys
import os

def get_bin_name(i):
	if (i == 0):
		return "CONTROL"
	
	name = ""
	if (i & (1 << 0)):
		name += "Q"
	if (i & (1 << 1)):
		name += "R"
	if (i & (1 << 2)):
		name += "T"
	if (i & (1 << 3)):
		name += "m"
	if (i & (1 << 4)):
		name += "r"
	if (i & (1 << 5)):
		name += "t"
	if (i & (1 << 6)):
		name += "n"
	
	return name
	
def get_directives(i):
	directives = ""
	if (i & (1 << 0)):
		directives += " -DQRYTEX=1 "
	else:
		directives += " -DQRYTEX=0 "
	
	if (i & (1 << 1)):
		directives += " -DREFTEX=1 "
	else:
		directives += " -DREFTEX=0  "
		
	if (i & (1 << 2)):
		directives += " -DTREETEX=1  "
	else:
		directives +=  "-DTREETEX=0  "
			
	if (i & (1 << 3)):
		directives += " -DMERGETEX=1  "
	else:
		directives += " -DMERGETEX=0  "
		
	if (i & (1 << 4)):
		directives += " -DREORDER_REF=1  "
	else:
		directives += " -DREORDER_REF=0  "
		
	if (i & (1 << 5)):
		directives += " -DREORDER_TREE=1  "
	else:
		directives += " -DREORDER_TREE=0  "
		
	if (i & (1 << 6)):
		directives += " -DRENUMBER_TREE=1  "
	else:
		directives += " -DRENUMBER_TREE=0  "
	
	return directives + '"\n'

def print_make_rules(r, file_name):
	f = open(file_name, "w")
	for i in range(0, r):
		bin_name = get_bin_name(i)
		dirs = get_directives(i)
		rule = '%s: clean\n\tmake all "BINNAME = %s" "COMMONFLAGS = ${COMMONFLAGS} ' % (bin_name, bin_name)
		rule += dirs
		print >> f, rule
		
		rule = '%s_cubin: clean\n\tmake %s.cubin "BINNAME = %s" "COMMONFLAGS = ${COMMONFLAGS} ' % (bin_name, bin_name, bin_name)
		rule += dirs
		print >> f, rule
		
def print_make_test_rule(r, file_name):
	make_test_rule = "test:"
	for i in range(0, r):
		bin = get_bin_name(i)
		make_test_rule += " " + bin
	make_test_rule += "\n.SUFFIXES : .cu .cu_dbg_o .c_dbg_o .cpp_dbg_o .cu_rel_o .c_rel_o .cpp_rel_o .cubin\n"
	
	make_test_cubin_rule = "test_cubin:"
	for i in range(0, r):
		bin = get_bin_name(i)
		make_test_cubin_rule += " " + bin + "_cubin"
	
	f = open(file_name, "w")
	print >> f, make_test_rule
	print >> f, make_test_cubin_rule
	
def print_bash_rules(r, file_name):
	f = open(file_name, "w")
#	print >> f, "#!/bin/bash"
#	print >> f, "$include runm-mummergpu.sh"
	
	for i in range(0, r):
		bin = get_bin_name(i)
		print >> f, "\trun_mummergpu %s $REF $QRY $MINMATCH $ORG" % (bin)
		
def main():
	configs = 128
	print_make_rules(configs, "rules.mk")
	print_make_test_rule(configs, "test_rule.mk")
	print_bash_rules(configs, "cmds.sh")
	


if __name__ == '__main__':
	main()

