# Implement Nearest Neighbour classifier here!
import numpy as np
import argparse
import os
import csv
import random
import math
import operator
from sklearn.metrics import f1_score



#Read Data from *.csv file : 
def read_csv_file(path,path_label,flag):
	#print 'Inside read csv file'
	#c=0
	#print path
	f = open(path,"r")
	data = []
	for row in f:
		#c+=1
		if flag == 0:
			row = row.split()
		elif flag == 1:
			row = row.split(",")
		#print row
		row = [ float(x) for x in row]
		data.append(row)
		
	#c=0
	f = open(path_label,"r")
	label = []
	for row in f:
		#c+=1
		if flag == 0:
			row = row.split()
		elif flag == 1:
			row = row.split(",")
		row = [ float(x) for x in row]
		label.append(row)
	
	return [np.asarray(data),np.asarray(label)]



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
	return [np.asarray(input_train),np.asarray(input_train_label), np.asarray(input_test),np.asarray(input_test_label)]
	

#*********************************** NN MODEL******************
    
#find class of a test example
def find_class(neighbours):
    class_count={}
    for row in range(len(neighbours)):
        #label = neighbours[row][-1]
        label = neighbours[row][0]
        #print 'Label : ',label
        if label in class_count:
            class_count[label] +=1
        else:
            class_count[label] = 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1),reverse=True)
    #print sorted_class_count
    return sorted_class_count[0][0]

def get_top_k_neighbours(k,input_train,test_vector,input_train_label):
	#print type(input_train)
	all_distance = np.linalg.norm(input_train - test_vector,axis=1)
	arg_sort = np.argsort(all_distance)
	neighbours = []
	for i in range(k):
		neighbours.append(input_train_label[arg_sort[i]])
		
	return neighbours
#*********************** END OF MODEL *******************************
#Define calculate test accuracy
def calculate_test_accuracy(input_test,predictions):
	correct = 0
	
	for row in range(len(input_test)):
		if input_test[row] == predictions[row]:
			correct +=1
	
	macro = f1_score(input_test, predictions, average='macro')
	micro = f1_score(input_test, predictions, average='micro')
	score = f1_score(input_test, predictions, average='weighted')
	return 100*(correct / float(len(input_test))),macro,micro,score

def main(csv_txt,write_flag,path,label_path):
	# Raed from csv file -
	if csv_txt == 0:
		input_data,input_label = read_csv_file(path,label_path,0)
		#input_label = read_csv_file(label_path,0)
	if csv_txt == 1:
		input_data,input_label = read_csv_file(path,label_path,1)
	
	
	#print input_data.shape
	#input_label = read_csv_file(label_path,0)
	split_ratio = 0.80
	
	input_train ,input_train_label, input_test,input_test_label = split_input_data(input_data,input_label,split_ratio)
	#input_train_classes = split_input_train_data_into_classes(input_train,input_train_label)
	print ('Train sample size : {0}, Test sample size : {1}').format(len(input_train),len(input_test))
	
	
	#Build model with NN
	
	predict_label=[]
	k=3
	
	for test_elem in range(len(input_test)):
		neighbours = get_top_k_neighbours(k, input_train, input_test[test_elem],input_train_label)
			
		cur_class = find_class(neighbours)
		predict_label.append(cur_class)
	
	
	#print predict_label
	#print input_test_label
	true_label = [float(input_test_label[i][0]) for i in range(len(input_test_label))]
	#print "Actual Test  :",true_label
	
	#Calculate the accuracy
	#test_accuracy , macro , micro , score = calculate_test_accuracy(input_test_label,predict_label)
	test_accuracy , macro , micro , score = calculate_test_accuracy(true_label,predict_label)
	
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
	
	root_path = "/home/sda/Desktop/TIPR/FIRST ASSIGNMENT/tipr-first-assignment-master/data/"
	
	csv_txt = 2
		
	if args.dolphins:
		csv_txt = 0
		write_flag=0
		path = os.path.join( root_path, "dolphins/dolphins.csv" )
		label_path = os.path.join( root_path, "dolphins/dolphins_label.csv" )
	if args.pubmed:
		write_flag=1
		csv_txt = 0
		print 'root_path',root_path
		path = os.path.join( root_path, "pubmed/pubmed.csv" )
		#print path
		print path
		label_path = os.path.join( root_path, "pubmed/pubmed_label.csv" )
		#print label_path
	if args.twitter:
		write_flag=2
		csv_txt = 1
		path = os.path.join( root_path, "twitter/feature_vector.csv")
		label_path = os.path.join( root_path, "twitter/twitter_label.txt")
	
	
	main(csv_txt,write_flag,path,label_path)
	
	
