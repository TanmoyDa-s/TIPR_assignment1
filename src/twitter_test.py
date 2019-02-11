import re
import numpy as np
import csv

word_dict = {}

def read_txt_file(path):
	f= open(path,"r")
	count =0;
	allsentence=[]
	for sentence in f:
		count += 1
		allsentence.append(sentence)
		#if count>5:
		#	break
	#print "total sentence : ",count
	return allsentence



def words_extract_from_sentence(sentence):
	#print "Input : ", sentence
	ignore = ['\n']
	words = re.sub("[^\w]"," ", sentence).split()
	#print "words : ",words
	words_lower = [w.lower() for w in words if w not in ignore and len(w)>=3]
	
	#print "words extract : " ,words_lower
	for w in words_lower:
		print w,len(w)
		if len(w) > 2:
			if w not in word_dict:
				word_dict[w] = 1
			else:
				word_dict[w] +=1
	return words_lower

def tokenize_sentence(allsentence):
	unique_words = []
	for sentence in allsentence:
		words = words_extract_from_sentence(sentence)
		unique_words.extend(words)
	unique_words = sorted(list(set(unique_words)))
	
	return unique_words

def generate_bag_of_words(allsentence):
	vocab = tokenize_sentence(allsentence)
	return vocab

def words_extract_for_feature_vector(sentence):
	#print "Input : ", sentence
	ignore = ['\n']
	words = re.sub("[^\w]"," ", sentence).split()
	#print "words : ",words
	words_lower = [w.lower() for w in words if w not in ignore and len(w)>=3]
	
	#print "words extract : " ,words_lower
	#for w in words_lower:
	#	print w,len(w)
	#	if len(w) > 2:
	#		if w not in word_dict:
	#			word_dict[w] = 1
	#		else:
	#			word_dict[w] +=1
	return words_lower


def create_feature_vector(vocab,sentence):
	words = words_extract_for_feature_vector(sentence)
	vector = np.zeros(len(vocab))
	for w in words:
		for i,word in enumerate(vocab):
			if word == w:
				vector[i] +=1
	return vector


#w=[]
#for s in allsentence:
#	print count,s
#	#print s.split()
#	words = re.sub("[^\w]"," ",s).split()
#	print "HELLO : ", words
#	w.extend(words)
#	print "HELLO : ", words
#	print "Length : ",len(words),"\n\n"
#	print "unsorted : ", w
#	print "Length : ",len(w),"\n\n"
#	w=sorted(list(set(w)))
#	print "HELLO sorted : ", w
#	print "Length : ",len(w),"\n\n"
	


if __name__ == '__main__':
	allsentence = read_txt_file("twitter.txt")
	print allsentence

	vocab = generate_bag_of_words(allsentence)
	print vocab
	print "size  : ", len(vocab)
	#rint word_dict
	
	
	print len(word_dict)
	feature_vector = []
	for sentence in allsentence:
		vector = create_feature_vector(vocab,sentence)
		#print sentence
		#print vector
		feature_vector.append(vector)
		
	feature_vector = np.asarray(feature_vector)
	print feature_vector
	count = 0
	with open("feature_vector.csv", "w") as writeFile:
		writer = csv.writer(writeFile)
		for row in range(feature_vector.shape[0]):
			#print "count : ",count
			writer.writerow(feature_vector[row,:])
			count +=1
	
	print "Total row : ",feature_vector.shape[0]
	print "Total col : ",feature_vector.shape[1]
	
