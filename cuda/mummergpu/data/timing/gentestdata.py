#!/usr/bin/env python
# encoding: utf-8
"""
divcost.py

Created by Cole Trapnell on 2008-02-24.
Copyright (c) 2008 Cole Trapnell. All rights reserved.
"""

import sys
import getopt


help_message = '''
Generates a query file and reference file that will measure the cost of 
divergence
'''


class Usage(Exception):
	def __init__(self, msg):
		self.msg = msg


def main(argv=None):
	if argv is None:
		argv = sys.argv
	try:
		try:
			opts, args = getopt.getopt(argv[1:], "ho:v", ["help", "output="])
		except getopt.error, msg:
			raise Usage(msg)
	
		# option processing
		for option, value in opts:
			if option == "-v":
				verbose = True
			if option in ("-h", "--help"):
				raise Usage(help_message)
			if option in ("-o", "--output"):
				output = value
	
		tree_nodes = 512
		
		# Head should should be n characters long, where n is the number of 
		# queries in a single warp.
		warp_size = 32
		head = warp_size
		
		# tail can as long as needed to make the tree as big as we can fit 
		# in the texture cache (512 nodes on a single multiprocessor)
		head_nodes = 2*head - 1
		tail_nodes = tree_nodes - head_nodes - 3
		tail = (tail_nodes + 1) / 2
		ref = []
		head = ''.join(['T' for i in range(0, head)])
		
		tail = ''.join(['C' for i in range(0, tail)])
		ref = head + 'G' + tail
		def print_as_fasta(f, defline, seq):
			print >> f, defline
			i = 0
			while i + 60 < len(seq):
				print >> f, seq[i:i + 60]
				i += 60
			print >> f, seq[i:]

		ref_file = open("divcostref.fa", "w")
		print_as_fasta(ref_file, ">divcostref", ref)
		
		blocks = 1024
		num_queries = warp_size * blocks
		
		# tail_qry_file = open("divcosttailqry.fa", "w")
		# for i in range(0, num_queries):
		# 	print_as_fasta(tail_qry_file, ">q" + str(i), tail)
		# 	
		# head_qry_file = open("divcostheadqry.fa", "w")
		# for i in range(0, num_queries):
		# 	print_as_fasta(head_qry_file, ">q" + str(i), head[(i % warp_size):] + 'A')
			
		
		queries = []
		for i in range(0, num_queries):
			queries.append( head[(i % warp_size):] + 'A' + tail ) 
			
		full_div_file = open("divcost_fulldivqry.fa", "w")
		for i in range(0, num_queries):
			print_as_fasta(full_div_file, ">q" + str(i), queries[i])
			
		queries.sort(lambda x,y: len(x) - len(y))	
		
		no_div_file = open("divcost_nodivqry.fa", "w")
		for i in range(0, num_queries):
			print_as_fasta(no_div_file, ">q" + str(i), queries[i])
		
		
	except Usage, err:
		print >> sys.stderr, sys.argv[0].split("/")[-1] + ": " + str(err.msg)
		print >> sys.stderr, "\t for help use --help"
		return 2


if __name__ == "__main__":
	sys.exit(main())
