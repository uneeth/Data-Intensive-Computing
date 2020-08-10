#!/usr/bin/env python3
import sys
for line in sys.stdin:
        #print(line)
        words=line.strip().split(";")
        #print(words, len(words))
        primaryKey = words[0].strip()
        for i in range(1, len(words)):
                words[i].strip()
                print('%s\t%s'%(primaryKey, (str(len(words)) + "." + str(i) + ":" + words[i])))

