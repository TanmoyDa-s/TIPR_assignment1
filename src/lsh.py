# Implement code for Locality Sensitive Hashing here!
import argparse
import os
import numpy as np
from cross_validation import read_csv_file
from cross_validation import split_input_data
from projections import generate_projection_matrix

def Generate_hash_function(number_hash_func,orginal_dim,k):
	hash_function = []
	for i in range(number_hash_func):
		random_matrix = generate_projection_matrix(orginal_dim,k)
		hash_function.append(random_matrix)
	
	return hash_function


def Generate_hash_code(projected_vector):
	bool_vector = (np.asarray(projected_vector) >0).astype('int')
	return ''.join(bool_vector.astype('str'))

def lsh(number_hash_func,orginal_dim,k,input_data,input_label):
	all_hash_function = Generate_hash_function(number_hash_func,orginal_dim,k)
	
	hash_tables=[]
	for i in range(number_hash_func):
		new_hash_table = {}
		list_hash_value =[]
		projected_data = np.dot(input_data , all_hash_function[i])
		for row in range(len(projected_data)):
			hash_code = Generate_hash_code(projected_data[row])
			list_hash_value.append(hash_code)
			if hash_code not in new_hash_table:
				new_hash_table[hash_code] = []
							
			new_hash_table[hash_code].append(list(projected_data[row]) + input_label[row])
		hash_tables.append(new_hash_table)
	return hash_tables,all_hash_function

def get_all_localiy_sensetive_elements(hash_tables,input_vector):
	hash_code = Generate_hash_code(input_vector)
	all_neighbours = []
	for i in range(len(hash_tables)):
		if hash_code in hash_tables[i]:
			#print "Total_neighbours : ",len(hash_tables[i][hash_code])
			return hash_tables[i][hash_code]
	#return all_neighbours



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
	
	# Raed from csv file -
	if csv_txt == 0:
		input_data, input_label = read_csv_file(path,label_path,0)
		#input_label = read_csv_file(label_path,0)
	if csv_txt == 1:
		input_data,input_label = read_csv_file(path,label_path,1)
	
	
	#print input_data.shape
	#input_label = read_csv_file(label_path,0)
	split_ratio = 0.80
	number_hash_func = 1
	print len(input_data)
	print len(input_data[0])
	orginal_dim = len(input_data[0])
	#k = int(orginal_dim/3)
	k=10
	hash_tables,all_hash_functions = lsh(number_hash_func,orginal_dim,k,input_data,input_label)
	
	all_neighbour = get_all_localiy_sensetive_elements(hash_tables,np.dot(input_data[10],all_hash_functions[0]))
	
	
	
	#print all_neighbour
	print len(all_neighbour)
	
