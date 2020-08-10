#!/usr/bin/env python3
from operator import itemgetter
import sys
from heapq import nlargest

vsPerk = {}
N = 10
word = None
triDict = {}
resDict = {}

for line in sys.stdin:
	word, count=line.split('\t')
	try:
		count = int(count)
	except ValueError:
		continue
	try:
		vsPerk[word].append(count)
	except KeyError:
		vsPerk[word] = [count]

for key,items in vsPerk.items():
	triDict[key] = sum(items)
res = nlargest(N, triDict, key = triDict.get) 

for key in res:
	print(str(key)+':'+str(triDict[key]))
sys.stdout.flush()
