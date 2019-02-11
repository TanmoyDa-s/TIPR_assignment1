# Implement Bayes Classifier here!
import numpy as np
import argparse
import os
import csv
import random
import math
from sklearn.metrics import f1_score


#Read Data from *.csv file : 
def read_csv_file(path,flag):
	#print 'Inside read csv file'
	c=0
	f = open(path,"r")
	data = []
	for row in f:
		#print "c : ",c
		c+=1
		if flag == 0:
			row = row.split()
		elif flag == 1:
			row = row.split(",")
		row = [ float(x) for x in row]
		data.append(row)
	
	return np.asarray(data)

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
	return [input_train, input_test,input_train_label,input_test_label]
	
#Define input data into classes
def split_input_train_data_into_classes(input_train,input_train_label):
	#print 'Inside split input train data into classes'
	input_train_classes = {}
	input_train_label = np.asarray(input_train_label)
	#print type(input_train), 'train'
	#print type(input_train_label), 'train_label'
	train_size = len(input_train)
	for index in range(train_size):
		current_sample = input_train[index]
		#print current_sample
		#print input_train_label[index][0]
		if(input_train_label[index][0] not in input_train_classes):
			input_train_classes[input_train_label[index][0]] = []
			
		input_train_classes[input_train_label[index][0]].append(current_sample)
		
	return input_train_classes 



#*********************************** BAYES MODEL******************
#Define mean_feature
def mean_feature(feature):
	#print 'Inside mean feature'
	SUM = sum(feature)
	TOTAL = float(len(feature))
	if TOTAL == 0:
		return 1
	return SUM/TOTAL

#Define standard deviation of a feature
def sd_feature(feature):
	#print 'Inside sd feature'
	mu = mean_feature(feature)
	squared_feature = [ pow( (x-mu),2 ) for x in feature ]
	sum_sqr = sum(squared_feature)
	TOTAL = float(len(feature)-1)
	if TOTAL == 0:
		return 1
	varience = sum_sqr /TOTAL
	sd = math.sqrt(varience)
	return sd


#Define calculate MLE for a class
def calculate_MLE(class_samples):
	#print 'Inside calculate MLE'
	mle_of_class = [ (mean_feature(feature),sd_feature(feature)) for feature in zip(*class_samples) ]
	
	return mle_of_class

#Define Bayes model classifier 		
def build_bayes_model_with_MLE(input_train_classes):
	#print 'Inside bild bayes model with MLE'
	model = {}
	
	for each_class_value,class_samples  in input_train_classes.iteritems():
		model[each_class_value] = calculate_MLE(class_samples)
		
	return model
#*********************** END OF MODEL *******************************

#*************************CODE FOR TEST RESULT PREDICTION *****************
#Define propability using normal pdf
def calculate_probability_normal_pdf(feature,mu,sd):
	#print 'Inside calculate probability normal pdf'
	temp1 = math.pow((feature-mu),2)
	temp2 = 2 * math.pow(sd,2)
	if temp2 == 0:
		return 1
	temp1 = math.exp(-(temp1)/temp2)
	temp2 = math.sqrt(2*math.pi)*sd
	if temp1 == 0:
		return 1
	return (1/temp2)*temp1

#Define probability in each class
def calculate_propability_for_each_class(model,test_sample):
	#print 'Inside calculate probability for each class'
	probability_in_each_class = {}
	
	for class_label,class_model_par in model.iteritems():
		probability_in_each_class[class_label] = 1
		for index in range(len(class_model_par)):
			mu,sd = class_model_par[index]
			feature = test_sample[index]
			probability_in_each_class[class_label] += math.log(calculate_probability_normal_pdf(feature,mu,sd))
			
	return probability_in_each_class

#Define prediction of each test sample
def predict_each_sample(model,test_sample):
	#print 'Inside predict each sample'
	probability_in_each_class = calculate_propability_for_each_class(model,test_sample)
	predict_label,prediction_probability = None,-1
	
	for class_label,class_probability in probability_in_each_class.iteritems():
		if predict_label is None or class_probability > prediction_probability:
			predict_label = class_label
			prediction_probability = class_probability
			
	return predict_label 

#Define test prediction
def predict_on_test_data(model,input_test):
	#print 'Inside predict on test data'
	test_prediction = []
	for index in range(len(input_test)):
		predict_result = predict_each_sample(model,input_test[index])
		test_prediction.append(predict_result)
	
	return test_prediction

#*********************** END OF PREDICTION CODE ************************************ 
#Define calculate test accuracy
def calculate_test_accuracy(input_test_label,test_prediction):
	#print 'Inside calculate test accuracy'
	correct_count = 0
	for row in range(len(input_test_label)):
		if input_test_label[row] == test_prediction[row]:
			correct_count +=1
	
	macro = f1_score(input_test_label, test_prediction, average='macro')
	micro = f1_score(input_test_label, test_prediction, average='micro')
	score = f1_score(input_test_label, test_prediction, average='weighted')
	return 100.0 * (correct_count/float(len(input_test_label))) , macro,micro,score


def main(csv_txt,write_flag,path,label_path):	
	# Raed from csv file -
	if csv_txt == 0:
		input_data = read_csv_file(path,0)
		#input_label = read_csv_file(label_path,0)
	if csv_txt == 1:
		input_data = read_csv_file(path,1)
	
	input_label = read_csv_file(label_path,0)
	split_ratio = 0.80
	#split the data into test and train set
	input_train , input_test , input_train_label, input_test_label = split_input_data(input_data,input_label,split_ratio)
	input_train_classes = split_input_train_data_into_classes(input_train,input_train_label)
	
	
	#Build model with MLE
	model = build_bayes_model_with_MLE(input_train_classes)
	
	
	#Predict the result for input_test data
	test_prediction = predict_on_test_data(model,input_test)
	
	#print len(input_test_label)
	#print len(test_prediction)
	
	#Calculate the accuracy
	test_accuracy,macro,micro,score = calculate_test_accuracy(input_test_label,test_prediction)
	
	
	print ('Test Accuracy :: {0}%\n\nTest Macro F1-score :: {1}\n\nTest Micro F1-score :: {2}\n\nTest weighted F1 score :: {3}').format(test_accuracy,macro*100,micro*100,score*100)





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser =argparse.ArgumentParser()
	parser.add_argument("-d","--dolphins",action = "store_true")
	parser.add_argument("-d","--dolphins_labels",action = "store_true")
	parser.add_argument("-p","--pubmed",action = "store_true")
	parser.add_argument("-p","--pubmed_labels",action = "store_true") 
	parser.add_argument("-t","--twitter",action = "store_true")
	args=parser.parse_args()
	
	#root_path=path_dir = "/home/sda/Desktop/TIPR/FIRST ASSIGNMENT/tipr-first-assignment-master/data/"
	csv_txt = 2
		
	if args.dolphins:
		csv_txt = 0
		write_flag=0
		path = os.path.join( root_path, "dolphins/dolphins.csv" )
		#path = os.path.join( root_path, "projected_data/dolphins/dolphins_2.csv" )
		label_path = os.path.join( root_path, "dolphins/dolphins_label.csv" )
	if args.pubmed:
		write_flag=1
		csv_txt = 0
		path = os.path.join( root_path, "pubmed/pubmed.csv" )
		print path
		label_path = os.path.join( root_path, "pubmed/pubmed_label.csv" )
		print label_path
	if args.twitter:
		write_flag=2
		csv_txt = 1
		path = os.path.join( root_path, "twitter/feature_vector.csv")
		label_path = os.path.join( root_path, "twitter/twitter_label.txt")
		
	main(csv_txt,write_flag,path,label_path)	
	# Raed from csv file -
	#if csv_txt == 0:
	#	input_data = read_csv_file(path,1)
	#	#input_label = read_csv_file(label_path,0)
	#if csv_txt == 1:
	#	input_data = read_csv_file(path,1)
	
	#input_label = read_csv_file(label_path,0)
	#split_ratio = 0.80
	#split the data into test and train set
	#input_train , input_test , input_train_label, input_test_label = split_input_data(input_data,input_label,split_ratio)
	#input_train_classes = split_input_train_data_into_classes(input_train,input_train_label)
	
	
	#Build model with MLE
	#model = build_bayes_model_with_MLE(input_train_classes)
	
	
	#Predict the result for input_test data
	#test_prediction = predict_on_test_data(model,input_test)
	
	#print len(input_test_label)
	#print len(test_prediction)
	
	#Calculate the accuracy
	#test_accuracy,macro,micro,score = calculate_test_accuracy(input_test_label,test_prediction)
	
	
	with open("Task_1_dolphins.txt", "a") as writeFile:
		writer = csv.writer(writeFile)
		data = []
		#data.append(2)
		data.append(len(input_data[0]))
		data.append(test_accuracy)
		#for row in range(feature_vector.shape[0]):
		#print "count : ",count
		writer.writerow(data)
		#count +=1
	
	
	#print ('Test Accuracy :: {0}%\n\nTest Macro F1-score :: {1}\n\nTest Micro F1-score :: {2}\n\nTest weighted F1 score :: {3}').format(test_accuracy,macro*100,micro*100,score*100)
