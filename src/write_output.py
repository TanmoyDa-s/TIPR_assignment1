import argparse
import os
import bayes
import nn
import cross_validation_NN as cv_nn
import cross_validation_bayes as cv_bayes
#import knn_CV_sklearn as bskl
import classifier_using_pca as cwp


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser =argparse.ArgumentParser()
	parser.add_argument("-d","--dolphins",action = "store_true")
	parser.add_argument("-p","--pubmed",action = "store_true") 
	parser.add_argument("-t","--twitter",action = "store_true")
	args=parser.parse_args()
	
	root_path=path_dir = "/home/sda/Desktop/TIPR/FIRST ASSIGNMENT/tipr-first-assignment-master/data/"
	csv_txt = 2
		
	if args.dolphins:
		csv_txt = 1
		write_flag=0
		number_test_file =17
		step =2
		path = os.path.join( root_path, "projected_data/dolphins/" )
		#path = os.path.join( root_path, "projected_data/dolphins/dolphins_2.csv" )
		label_path = os.path.join( root_path, "dolphins/dolphins_label.csv" )
	if args.pubmed:
		write_flag=1
		csv_txt = 1
		number_test_file = 55
		step =4
		path = os.path.join( root_path, "projected_data/pubmed/" )
		print path
		label_path = os.path.join( root_path, "pubmed/pubmed_label.csv" )
		print label_path
	if args.twitter:
		write_flag=2
		csv_txt = 1
		number_test_file =1499
		step = 90
		path = os.path.join( root_path, "projected_data/twitter/")
		label_path = os.path.join( root_path, "twitter/twitter_label.txt")
	
	
	for i in range(2,number_test_file,step):
		print 'current dim : ',i
		if write_flag == 0:
			f_name  = "dolphins_"+str(i)+".csv"
		elif write_flag == 1:
			f_name  = "pubmed_"+str(i)+".csv"
		elif write_flag ==2:
			f_name  = "twitter_"+str(i)+".csv"
			
		actual_path = os.path.join( path, f_name)
		print path
		#label_path = os.path.join( root_path, "twitter/twitter_label.txt")
		
		cv_nn.main(csv_txt,write_flag,actual_path,label_path)	
