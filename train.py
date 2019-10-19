from decision_tree import *
import numpy as np

# Load the datasets
filepath = ''
filename = 'clean_dataset.txt'
data = np.loadtxt(filepath+filename)

x = data[:,:-1]
y = data[:,-1]


def evaluate(predicted_label, true_label):
	"""
	Evaluate accuracy on predicted label given true label
	
	Args:
		predicted_label : 
		true_label      :
	Outputs: accuracy of the predicted label
	"""
	assert(len(predicted_label)==len(true_label))
	data_num = len(true_label)
	accuracy = float(sum(predicted_label == true_label))/data_num
	return accuracy


def compute_confusion_matrix(tree,test_data,num_label):
	"""
	compute confusion matrix of a learnt tree
	
	Args:
		tree	 (dict): learnt decision tree
		test_data (int): test data with label on the last column
		num_label (int): number of label category in dataset(row/col number of matrix)
	Outputs:
	"""
	#assert(type(tree)==dict)
	assert(type(test_data)==np.ndarray)
	assert(len(test_data.shape)==2 and test_data.shape[1]>1)
	assert(type(num_label)==int)
	assert(num_label>0)

	confusion_matrix=np.zeros([num_label,num_label])

	predict_label = tree.classify(data[:,:-1])
	gt_label = data[:,-1]
	for p, g in zip(predict_label, gt_label):
		confusion_matrix[int(p)-1,int(g)-1]+=1

	return confusion_matrix

tree = decision_tree()
tree.train(x,y)

y_pred = tree.classify(x)
print("Accuracy: " + str(evaluate(y_pred, y)))

confusion_matrix = compute_confusion_matrix(tree, data, len(set(y)))
print(confusion_matrix)
