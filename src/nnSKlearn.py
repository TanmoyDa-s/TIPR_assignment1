#Implement kNN Classifier using sklearn--

import argparse
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from cross_validation import read_csv_file
from cross_validation import split_input_data
from sklearn import metrics

def main(csv_txt,write_flag,path,label_path):
	# Raed from csv file -
	if csv_txt == 0:
		input_data, input_label = read_csv_file(path,label_path,0)
		#input_label = read_csv_file(label_path,0)
	if csv_txt == 1:
		input_data,input_label = read_csv_file(path,label_path,1)
	
	
	#print input_data.shape
	#input_label = read_csv_file(label_path,0)
	split_ratio = 0.80
	
	
	input_train, input_train_label, input_test,input_test_label = split_input_data(input_data,input_label,split_ratio)
	#input_train_classes = split_input_train_data_into_classes(input_train,input_train_label)
	print ('Train sample size : {0}, Test sample size : {1}').format(len(input_train),len(input_test))
	
	
	#bayes_model = GaussianNB()
	kNN_model = KNeighborsClassifier(n_neighbors=3)
	kNN_model.fit(input_train,input_train_label)
	test_prediction = []
	for i in range(len(input_test)):
		test_predicted= kNN_model.predict(np.atleast_2d(input_test[i]))
		print test_predicted
		test_prediction.append(test_predicted)
	
	print test_prediction
	
	print("Accuracy:",metrics.accuracy_score(input_test_label, test_prediction)*100,"%")





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
	
