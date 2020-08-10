#!/usr/bin/env python3
import sys
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

punctuations = '’“”`^-|,!()-[]{};:\\,<>.?@#$%^&*-=+~_—\'\"'
for line in sys.stdin:
	doc_id = os.environ["map_input_file"]
	doc_name = os.path.split(doc_id)[-1]
	words=line.strip().lower().split()
	for word in words:
		for x in word:
			if x in punctuations:
				word=word.replace(x,"")
		if word in stop_words:
			continue
		word = lemmatizer.lemmatize(word)

		print('%s\t%s'%(word, doc_name))
