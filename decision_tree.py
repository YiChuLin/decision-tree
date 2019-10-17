"""
decision_tree.py
"""
# Import the required libraries
import numpy as np
import matplotlib

# Load the datasets
filepath = 'wifi_db/'
filename = 'clean_dataset.txt'

data = np.loadtxt(filepath+filename)

def entropy(data):
	"""
	Usage: H = entropy(dataset)
	Description: This function calculates the entropy given a data set assuming the last column represents the labels
	Input: 2d numpy array
	Output: single value representing the entropy
	"""
	labels = set(data[:,-1])
	data_num = data.shape[0]
	instances = dict.fromkeys(labels,0)
	for d in data:
		instances[d[-1]] += 1
	prob = [float(k)/data_num for k in list(instances.values())]
	sep_entropy = [p*np.log2(p) for p in prob]
	return -sum(sep_entropy)

def find_split(data):
	"""
	Usage:
	Description:
	Outputs:
	"""
	attr_num = data.shape[1] - 1
	data_num = data.shape[0]
	info_gain = None
	node_split = None
	node_attr = None
	l_set = np.empty([0, data.shape[1]])
	r_set = np.empty([0, data.shape[1]])
	for attr in range(attr_num):
		# Find the optimal split in this attribute
		data = data[data[:,attr].argsort()]
		for i in range(data_num - 1):
			if data[i, attr] == data[i+1,attr]:
				continue
			else:
				split = (data[i+1, attr] + data[i,attr])/2.0
				#split the data based on the calculated split
				b1 = data[data[:,attr] > split]
				b2 = data[data[:,attr] <= split]
				# Calculate the information gain with this split
				# For simplicity we calculate the remainder and take the minus sign to quantify it
				Hb1 = entropy(b1)
				Hb2 = entropy(b2)
				neg_remainder = -(float(b1.shape[0])/data_num*Hb1 + float(b2.shape[0])/data_num*Hb2)
				if  info_gain == None or neg_remainder > info_gain:
					info_gain = neg_remainder
					node_split = split
					node_attr = attr
					l_set = b1
					r_set = b2
	return node_attr, node_split, l_set, r_set

def decision_tree_learning(sub_data, d = 0):
	"""
	Usage:
	Description: The Tree is represented in a recurrsive dictionary
	Inputs:
	Outputs:
	"""
	#Create Empty Dictionary
	node = {'attr':None, 'value':None, 'left':None, 'right':None}
	#Check if all labels are the same
	label_num = len(set(sub_data[:,-1]))
	if label_num == 1:
		# No further seperation is needed, return the node as leaf
		node['attr'] = 'leaf'; node['value'] = sub_data[0,-1]
		return node, d
	elif label_num == 0:
		node['attr'] = 'leaf'
		return node, d		
	else:
		attr, split,l_set, r_set = find_split(sub_data)
		node['attr'] = attr
		node['split'] = split
		node['left'], d1 = decision_tree_learning(l_set, d = d+1)
		node['right'], d2 = decision_tree_learning(r_set, d = d+1)
		depth = max(d1,d2)
		return node, depth

def classify(tree, data_point):
	at_leaf = False
	label = None
	while not at_leaf:
		attr = tree['attr']
		if attr == 'leaf':
			label = tree['value']
			at_leaf = True
		else:
			split = tree['split']
			if data_point[attr] > split:
				tree = tree['left']
			else:
				tree = tree['right']
	return label

def evaluate(tree, data):
	data_num = data.shape[0]
	correct_num = 0
	for data_point in data:
		prediction = classify(tree, data_point)
		correct_num += (prediction == data_point[-1])
	accuracy = float(correct_num)/data_num
	return accuracy

tree, d = decision_tree_learning(data)
accuracy=evaluate(tree, data)
print(str(accuracy))