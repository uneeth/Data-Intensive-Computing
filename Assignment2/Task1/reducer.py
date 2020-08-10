#!/usr/bin/env python3
from operator import itemgetter
import sys

vsPerk= dict()

for line in sys.stdin:
	word, count = line.split('\t')
	try:
		count = int(count)
	except ValueError:
		continue
	try:
		vsPerk[word].append(count)
	except KeyError:
		vsPerk[word]=[count]

for key,items in vsPerk.items():
	print(key,sum(items))

sys.stdout.flush()
