#!/usr/bin/env python3
import sys

for line in sys.stdin:
	words, count=line.strip().lower().split(":")
	print("%s\t%s"%(words,count))
