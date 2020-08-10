#!/usr/bin/env python3
from operator import itemgetter
import sys

vsPerk= dict()

for line in sys.stdin:
	word, document = line.split('\t')
	try:
		vsPerk[word].append(document[:-1])
	except KeyError:
		vsPerk[word]=[document[:-1]]

for key,items in vsPerk.items():
	print(key,":",set(items))

sys.stdout.flush()
