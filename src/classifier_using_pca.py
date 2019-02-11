
# Implement cross validation here
# Implement Bayes Classifier here!
import numpy as np
import argparse
import os
import csv
import random
import math
import operator
#import bayes
import nn
import lsh
from sklearn.decomposition import PCA
#import cross_validation as CV



#Read Data from *.csv file : 
# ******************************* READ FILE **************************
def read_csv_file(path,path_label,flag):
	# Read the data file.
	f = open(path,"r")
	data = []
	for row in f:
		if flag == 0:
			row = row.split()
		elif flag == 1:
			row = row.split(",")
		row = [ float(x) for x in row]
		data.append(row)
	
	# Read the label file
	f = open(path_label,"r")
	label = []
	for row in f:
		if flag == 0:
			row = row.split()
		elif flag == 1:
			row = row.split(",")
		row = [ float(x) for x in row]
		label.append(row)
		
	# Megre data and label file.
	#for index in range(len(data)):
	#	data[index].append(label[index][0])
	return [data, label]


# ******************************* SPLIT TEST & TRAIN **************************
#Define split test train function
def split_input_data(input_data,input_label,split_ratio):
	#print 'Inside split input data'
	input_train = []
	input_train_label = []
	
	input_test = list(input_data)
	input_test_label = list(input_label)
	current_train_size = 0
	actual_train_size = int(len(input_data)*split_ratio)
	
	while current_train_size < actual_train_size:
		index = random.randrange(len(input_test))
		input_train.append(input_test.pop(index))
		input_train_label.append(input_test_label.pop(index))
		current_train_size = len(input_train)
	return [input_train,input_train_label, input_test,input_test_label]
	
# *************************** CREATE FOLD FOR CROSS VALIDATION********************************

def classifier_with_pca(pca_dim,k,input_train,input_test,input_train_label):
	predict_label=[]
	#k=3
	pca =PCA(n_components = pca_dim)
	input_train = pca.fit_transform(input_train)
	input_test = pca.fit_transform(input_test)
	
	
	for test_elem in range(len(input_test)):
		neighbours = nn.get_top_k_neighbours(k, input_train, input_test[test_elem],input_train_label)
			
		cur_class = nn.find_class(neighbours)
		predict_label.append(cur_class)
		
	return predict_label
	

def main(csv_txt,write_flag,path,label_path):
	# Raed from csv file -
	if csv_txt == 0:
		input_data, input_label = read_csv_file(path,label_path,0)
		#input_label = read_csv_file(label_path,0)
	if csv_txt == 1:
		input_data,input_label = read_csv_file(path,label_path,1)
	
	split_ratio = 0.80
	
	input_train, input_train_label, input_test,input_test_label = split_input_data(input_data,input_label,split_ratio)
	k=3
	
	number_hash_function = 1
	#print len(input_data)
	#print len(input_data[0])
	orginal_dim = len(input_data[0])
	#k = int(orginal_dim/3)
	pca_dim=10
	predict_label = classifier_with_pca(pca_dim,k,input_train,input_test,input_train_label)
	
	
	input_test_label = [float(input_test_label[i][0]) for i in range(len(input_test_label))]
	#print predict_label
	#print input_test_label
	test_accuracy , macro , micro , score = nn.calculate_test_accuracy(input_test_label,predict_label)
	#test_accuracy= calculate_test_accuracy(true_label,predict_label)
	
	#print ('Test Accuracy :: {0}% for k = {1}').format(test_accuracy,k)
	print ('Test Accuracy          :: {0} %\n\nTest Macro F1-score    :: {1}\n\nTest Micro F1-score    :: {2}\n\nTest weighted F1 score :: {3}').format(test_accuracy,macro*100,micro*100,score*100)





#=======================================>> MAIN <<===========================
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser =argparse.ArgumentParser()
	parser.add_argument("-d","--dolphins",action = "store_true")
	parser.add_argument("-p","--pubmed",action = "store_true") 
	parser.add_argument("-t","--twitter",action = "store_true")
	args=parser.parse_args()
	
	root_path=path_dir = "/home/sda/Desktop/TIPR/FIRST ASSIGNMENT/tipr-first-assignment-master/data/"
	#root_path=path_dir = "/data/"
	csv_txt = 2
		
	if args.dolphins:
		csv_txt = 0
		write_flag=0
		path = os.path.join( root_path, "dolphins/dolphins.csv" )
		label_path = os.path.join( root_path, "dolphins/dolphins_label.csv" )
	if args.pubmed:
		write_flag=1
		csv_txt = 0
		path = os.path.join( root_path, "pubmed/pubmed.csv" )
		#print path
		label_path = os.path.join( root_path, "pubmed/pubmed_label.csv" )
		#print label_path
	if args.twitter:
		write_flag=2
		csv_txt = 1
		path = os.path.join( root_path, "twitter/feature_vector.csv")
		label_path = os.path.join( root_path, "twitter/twitter_label.txt")
	
	
	
	main(csv_txt,write_flag,path,label_path)
	
