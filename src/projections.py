# Implement code for random projections here!

import numpy as np
import argparse
import os
import csv

#import /home/sda/Desktop/TIPR/FIRST ASSIGNMENT/tipr-first-assignment-master/data/twitter/twitter_test
#import twitter_test
# Function to generate the random projection matrix from normal distribution
def generate_projection_matrix(d,k):
	mu, sigma = 0,1
	projection_matrix = np.random.normal(mu,sigma,d*k)
	return projection_matrix.reshape(d,k)
	

# Function to generate low dimentional data set	
def calculate_low_dimension_data(input_data,projection_matrix):
	return np.dot(input_data,projection_matrix)


#Read Data from *.csv file : 
def read_csv_file(path,flag):
	c=0
	f = open(path,"r")
	data = []
	for row in f:
		print "c : ",c
		c+=1
		if flag == 0:
			row = row.split()
		elif flag == 1:
			row = row.split(",")
		row = [ float(x) for x in row]
		data.append(row)
	
	return np.asarray(data)

#Write  Data into *.csv file for forther use.
def write_csv_file(path,data):
	with open(path, "w") as writeFile:
		writer = csv.writer(writeFile)
		for row in range(data.shape[0]):
			writer.writerow(data[row,:])
	
	writeFile.close()
		
def main(csv_txt,write_flag,path,label_path):		
	if csv_txt == 0:
		# Raed from csv file -
		input_data = read_csv_file(path,0)
	if csv_txt == 1:
		input_data = read_csv_file(path,1)
		
	total_example = input_data.shape[0]
	orginal_dim = input_data.shape[1]
	print "Orginal Shape : ",input_data.shape
		
	# Create projection matrix -
	for k in range(2,orginal_dim/2+1,2):
		projection_matrix = generate_projection_matrix(orginal_dim,k)
		projected_data = calculate_low_dimension_data(input_data,projection_matrix)
			
		print "k = ", k
		print "projection matrix Shape : ", projection_matrix.shape
		print "projected data Shape : ", projected_data.shape
			
			
		if(write_flag==0):
			file_name = "projected_data/dolphins/dolphins_"+str(k)+".csv"
			matrix_file_name = "projected_matrix/dolphins/proj_matrix_dim_"+str(k)+".csv"
		elif(write_flag==1):
			matrix_file_name = "projected_matrix/pubmed/proj_matrix_dim_"+str(k)+".csv"
			file_name = "projected_data/pubmed/pubmed_"+str(k)+".csv"
		elif(write_flag==2):
			matrix_file_name = "projected_matrix/twitter/proj_matrix_dim_"+str(k)+".csv"
			file_name = "projected_data/twitter/twitter_"+str(k)+".csv"
				
		write_file = os.path.join( root_path, file_name)
			
		write_csv_file(write_file,projected_data)
			
		write_file = os.path.join( root_path, matrix_file_name)
		write_csv_file(write_file,projection_matrix)
		print "write Complete for k : ",k ,"\n\n\n"


	print "Complete main "




if __name__=='__main__':
	parser =argparse.ArgumentParser()
	parser.add_argument("-d","--dolphins",action = "store_true")
	parser.add_argument("-p","--pubmed",action = "store_true") 
	parser.add_argument("-t","--twitter",action = "store_true")
	args=parser.parse_args()
	
	root_path=path_dir = "/home/sda/Desktop/TIPR/FIRST ASSIGNMENT/tipr-first-assignment-master/data/"
	csv_txt = 2
		
	
	if args.dolphins:
		csv_txt = 0
		write_flag=0
		path = os.path.join( root_path, "dolphins/dolphins.csv" )
	if args.pubmed:
		write_flag=1
		csv_txt = 0
		path = os.path.join( root_path, "pubmed/pubmed.csv" )
	if args.twitter:
		write_flag=2
		csv_txt = 1
		path = os.path.join( root_path, "twitter/feature_vector.csv")
		
	main(csv_txt,write_flag,path)
		
	
