import projections as proj
import os
import numpy as np
import csv
import argparse
import bayes
import nn
import classifier_using_lsh as cul
import classifier_using_pca as pca
import twitter_test as tt



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser =argparse.ArgumentParser()
	parser.add_argument('-test_file','-test_file',help='data path',required='false')
	parser.add_argument("-test_file_label","--test_file_label",required='false')
	parser.add_argument("-dataset","--dataset",required='false')
	args=parser.parse_args()
	
	
	
	v = vars(parser.parse_args())
	
	print v
	
	path = v["test_file"]
	path_label = v["test_file_label"]
	cs=0
	if v["dataset"] == 'twitter' or v["dataset"] == 'Twitter':
		cs = 1
		allsentence = tt.read_txt_file(path)
		print allsentence

		vocab = tt.generate_bag_of_words(allsentence)
		print vocab
		print "size  : ", len(vocab)
		#rint word_dict
	
	
		print len(word_dict)
		feature_vector = []
		for sentence in allsentence:
			vector = tt.create_feature_vector(vocab,sentence)
			#print sentence
			#print vector
			feature_vector.append(vector)
		
		feature_vector = np.asarray(feature_vector)
		with open("feature_vector.csv", "w") as writeFile:
			writer = csv.writer(writeFile)
			for row in range(feature_vector.shape[0]):
				#print "count : ",count
				writer.writerow(feature_vector[row,:])
				count +=1
		
		path = "feature_vector.csv"
	
	print "Bayes Classifier :"
	bayes.main(cs,1,path,path_label)
	print "\n\nknn :"
	nn.main(cs,1,path,path_label)
	print "\n\nClassifier with LSH :"
	cul.main(cs,1,path,path_label)
	print "\n\nClassifier with PCA :"
	pca.main(cs,1,path,path_label)	
	
	
	
