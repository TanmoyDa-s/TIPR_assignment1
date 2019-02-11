import matplotlib.pyplot as plt
import numpy as np


f = open("Task_3_bayes_twitter.txt","r")
data = []
flag=1
for row in f:
	if flag == 0:
		row = row.split()
	elif flag == 1:
		row = row.split(",")
	
	print row
	row = [ float(x) for x in row]
	data.append(row) 
		
data = np.asarray(data)

x1 = data[:,0]
y1 = data[:,1]


plt.plot(x1, y1, label = "bayes ")
plt.scatter(x1, y1, label= "bayes", color= "green",  marker= "*", s=30) 



f = open("Task_3_nn_twitter.txt","r")
data = []
for row in f:
	if flag == 0:
		row = row.split()
	elif flag == 1:
		row = row.split(",")
	print row
	row = [ float(x) for x in row]
	data.append(row) 
		
data = np.asarray(data)

x2 = data[:,0]
y2 = data[:,1]


plt.plot(x2, y2, label = "knn")

plt.scatter(x2, y2, label= " knn", color= "knn", marker= "*", s=30) 

#f = open("Task_2_nn_twitter.txt","r")
#data = []
#for row in f:
#	if flag == 0:
#		row = row.split()
#	elif flag == 1:
#		row = row.split(",")
#	row = [ float(x) for x in row]
#	data.append(row) 
		
#data = np.asarray(data)

#x3 = data[:,0]
#y3 = data[:,1]


#plt.plot(x3, y3, label = "knn twitter")


# naming the x axis 
plt.xlabel('Change of dimension') 
# naming the y axis 
plt.ylabel('Test accuracy') 


# giving a title to my graph 
plt.title('Task 3 twitter data') 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show()

