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


def cross_validate(data,k_fold=10):
	"""
	perform k-fold cross validation on learn_decision_tree
	
	Args:
		data (np.array): train/val data with label on the last column
		k	 (int): data.shape[0]>=k>0
	Outputs:
	"""
	assert(type(data)==np.ndarray)
	assert(len(data.shape)==2 and data.shape[1]>1)

	num_row=data.shape[0]
	assert(type(k_fold)==int)
	assert(k_fold>0 and k_fold<=num_row)

	np.random.shuffle(data)

	k_range = lambda k_head,k_tail: (num_row*np.arange(k_head,k_tail)/k_fold).astype(int)

	for k in range(0,k_fold):
		# divide data to train/val set
		val_data=data[k_range(k,k+1)]
		train_data=data[np.concatenate(k_range(0,k),k_range(k+1,k_fold))]

		# train decision tree
		tree,_=decision_tree_learning(train_data)
		acc=evaluate(tree,val_data)
		
		yield acc


def compute_confusion_matrix(tree,test_data,num_label):
	"""
	compute confusion matrix of a learnt tree
	
	Args:
		tree	 (dict): learnt decision tree
		test_data (int): test data with label on the last column
		num_label (int): number of label category in dataset(row/col number of matrix)
	Outputs:
	"""
	assert(type(tree)==dict)
	assert(type(test_data)==np.ndarray)
	assert(len(test_data.shape)==2 and test_data.shape[1]>1)
	assert(type(num_label)==int)
	assert(num_label>0)

	confusion_matrix=np.zeros([num_label,num_label])

	for data_point in test_data:
		predict_label=classify(tree,data_point)
		gt_label=data_point[-1]
		confusion_matrix[int(predict_label)-1,int(gt_label)-1]+=1

	return confusion_matrix
	

def main():
	tree, _ = decision_tree_learning(data)
	print(compute_confusion_matrix(tree,data,4))

if __name__=="__main__":
	main()
