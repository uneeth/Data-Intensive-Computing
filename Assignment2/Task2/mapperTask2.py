#!/usr/bin/env python3
import nltk
import re
import sys
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() 
punctuations = '’“”`^-|,!()-[]{};:\\,<>.?@#$%^&*-=+~_—\'\"'
result=[]
out=[]
stop_words = set(stopwords.words('english'))
for line in sys.stdin:
	words=line.strip().lower().split()
	for word in words:
		for x in word:
			if x in punctuations:
				word=word.replace(x,"")
		if word in stop_words:
			continue

		word = lemmatizer.lemmatize(word)

		if word == 'science':
			out.append('$1')
		elif word == 'fire':
			out.append('$2')
		elif word == 'sea':
			out.append('$3')
		else:
			out.append(word)

dictCheck = {'$1': 'science', '$2': 'fire', '$3': 'sea'}

for i in range(0, len(out)):
	if i+3 >= len(out):
		break;

	inter = str(out[i])+'_'+str(out[i+1])+'_'+str(out[i+2])

	if(inter.count('$')==2):
		
		if '$' in str(out[i]):
			inter1 = (str(dictCheck[out[i]])+'_'+str(out[i+1])+'_' + str(out[i+2]))
			inter1 = re.sub(r'\$.','$',inter1)
			result.append(inter1)

		if '$' in str(out[i+1]):
			inter2 = (str(out[i])+'_'+str(dictCheck[out[i+1]])+'_'+str(out[i+2]))
			inter2 = re.sub(r'\$.','$',inter2)
			result.append(inter2)
		
		if '$' in str(out[i+2]):
			inter3 = (str(out[i])+'_'+str(out[i+1])+'_'+str(dictCheck[out[i+2]]))
			inter3 = re.sub(r'\$.','$', inter3)
			result.append(inter3)
	else:
		inter4 = (str(out[i])+'_'+str(out[i+1])+'_'+str(out[i+2]))
		inter4 = re.sub(r'\$.', '$', inter4)
		result.append(inter4)

for i in result:
	if '$' in i:
		print('%s\t%s'%(i, 1))
