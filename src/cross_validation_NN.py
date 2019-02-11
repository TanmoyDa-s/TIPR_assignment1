# Implement cross validation here
# Implement Bayes Classifier here!
import numpy as np
import argparse
import os
import csv
import random
import math
import operator
import bayes
import nn
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
# Define cross-validation function
def create_fold(input_train,input_train_label,fold_num):
	fold_data = list()
	fold_label = list()
	
	data = list(input_train)
	data_label = list(input_train_label)
	
	fold_size = int(len(data)/fold_num)
	for i in range(fold_num):
		new_fold = list()
		new_fold_label = list()
		while len(new_fold) < fold_size:
			index = random.randrange(len(data))
			new_fold.append(list(data.pop(index)))
			new_fold_label.append(list(data_label.pop(index)))
		#print new_fold
		#print 'total element  : ' ,len(new_fold)
		fold_data.append(new_fold)
		fold_label.append(new_fold_label)
	return [fold_data,fold_label]


# ******************************* join foin for test and train **************************
def join_train_test_CV(i,fold_size,fold_data,fold_label):
	#train_data,train_data_label = set_train_test_CV(i,fold_size,fold_data,fild_data_labal)
	count = 0
	for j in range(fold_size):
		if j != i:
			if count == 0:
				#c=a[j]
				train_data = np.asarray(fold_data[j])
				train_data_label = np.asarray(fold_label[j])
			else:
				train_data=np.vstack((train_data,np.asarray(fold_data[j])))
				train_data_label=np.vstack((train_data_label,np.asarray(fold_label[j])))
				#c=np.vstack((c,a[j]))
			count +=1
	
	return train_data,train_data_label
	
# ******************************* Return Cross Validation model after cross validation process **************************
def cross_validation_process_nn(k,fold_size,fold_data,fold_label):
	#model_collection = {}
	for i in range(fold_size):
		#for i in range(len(a)):
		
		input_train,input_train_label = join_train_test_CV(i,fold_size,fold_data,fold_label)
				
		input_train = np.asarray(input_train)
		input_test = np.asarray(fold_data[i])
		input_test_label = np.asarray(fold_label[i])
		
		
		predict_label = []
		for test_elem in range(len(input_test)):
			neighbours = nn.get_top_k_neighbours(k, input_train, input_test[test_elem],input_train_label)
			
			cur_class = nn.find_class(neighbours)
			predict_label.append(cur_class)
	
		test_accuracy , macro , micro , score = nn.calculate_test_accuracy(input_test_label,predict_label)
		
		print ('Test Accuracy in iteration : {0} :: {1} %').format(i,test_accuracy)
	

def	main(csv_txt,write_flag,path,label_path):
	# Raed from csv file -
	if csv_txt == 0:
		input_data, input_label = read_csv_file(path,label_path,0)
		#input_label = read_csv_file(label_path,0)
	if csv_txt == 1:
		input_data,input_label = read_csv_file(path,label_path,1)
	
	split_ratio = 0.80
	
	input_train, input_train_label, input_test,input_test_label = split_input_data(input_data,input_label,split_ratio)
	k=3
	fold_size = 5
	random.seed(1)
	fold_data,fold_label = create_fold(input_train,input_train_label,fold_size)
	
	cross_validation_process_nn(k,fold_size,fold_data,fold_label)
	
	
	#print type(input_train)
	predict_label = []
	input_train = np.asarray(input_train)
	input_test = np.asarray(input_test)
	for test_elem in range(len(input_test)):
		neighbours = nn.get_top_k_neighbours(k, input_train, input_test[test_elem],input_train_label)
		cur_class = nn.find_class(neighbours)
		predict_label.append(cur_class)
	
	input_test_label = [float(input_test_label[row][0]) for row in range(len(input_test_label))]
	#print predict_label
	
	test_accuracy , macro , micro , score = nn.calculate_test_accuracy(input_test_label,predict_label)
	
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
	
