#!/usr/bin/env python3
from operator import itemgetter
import sys
from collections import Counter
from heapq import heappop, heappush, heapify, heappushpop 
  
# Creating empty heap 
heap = [] 
heapify(heap) 

values=[]
current_key=""
key=""

for line in sys.stdin:
	#line=line.strip()
        #print("Line:combiner ", line)
        key,value, category=line.split('\t')
        key = key.strip()
        value = value.strip()
        category = category.strip()
        if current_key==key:
            if len(heap) <= 5:
                heappush(heap, (-1*float(value), key, category))
            else:
                heappushpop(heap, (-1*float(value), key, category))
        else:
            if current_key != "":
                ans = []
                for i in list(heap):
                    ans.append(i[2])
                counter = Counter(ans)
                cat_pridict = counter.most_common(1)[0][0]
                print('%s\t%s' %(current_key, cat_pridict))
            heap = []
            heappush(heap, (-1*float(value), key, category))
            current_key=key
if current_key==key:
    ans = []
    for i in list(heap):
        ans.append(i[2])
    counter = Counter(ans)
    cat_pridict = counter.most_common(1)[0][0]
    print('%s\t%s' %(current_key, cat_pridict))
sys.stdout.flush()

