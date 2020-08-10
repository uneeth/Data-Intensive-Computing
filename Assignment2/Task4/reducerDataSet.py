#!/usr/bin/env python3
from operator import itemgetter
import sys

values=[]
current_key=""
key=""

for line in sys.stdin:
	#line=line.strip()
        #print("Line: ", line)
        key,value=line.split('\t')
        key = key.strip()
        value = value.strip().split(":")
        if current_key==key:
            values.append([value[0], value[1]])
        else:
            if current_key != "":
                ans =[]
                values.sort(key = lambda x:x[0])
                for i in values:
                    ans.append(i[1])
                print('%s\t%s' % (current_key,ans))
            values = []
            values.append([value[0], value[1]])
            current_key=key
if current_key==key:
    ans =[]
    values.sort(key = lambda x:x[0])
    for i in values:
        ans.append(i[1])
    print('%s\t%s' % (current_key,ans))
sys.stdout.flush()

