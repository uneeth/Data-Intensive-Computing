#!/usr/bin/env python3
import sys
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
punctuations = '’“”`^-|,!()-[]{};:\\,<>.?@#$%^&*-=+~_—\'\"'
for line in sys.stdin:
	words=line.strip().lower().split()
	for word in words:
		word.strip()
		for x in word:
			if x in punctuations:
				word=word.replace(x,"")
		word = lemmatizer.lemmatize(word)
		print('%s\t%s'%(word, 1))
		
