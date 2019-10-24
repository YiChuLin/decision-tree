import sys

import numpy as np

from decision_tree import *


def evaluate(predicted_label, true_label):
    """
    Evaluate accuracy on predicted label given true label

    Args:
            predicted_label : 
            true_label      :
    Outputs: accuracy of the predicted label
    """
    assert(len(predicted_label) == len(true_label))
    data_num = len(true_label)
    accuracy = float(sum(predicted_label == true_label))/data_num
    return accuracy


def tree_cross_validate(train_val_data, k_fold=10):
    """
    perform k-fold cross validation on decision tree
	iterable

    Args:
            train_val_data (np.array): train and val data with label on the last column
            k	 (int): data.shape[0]>=k>0
    Outputs:
			tree (class tree): the trained decision tree on each fold
			confusino matrix: the confusion matrix on each validation split
    """
    assert(type(train_val_data) == np.ndarray)
    assert(len(train_val_data.shape) == 2 and train_val_data.shape[1] > 1)

    num_row = train_val_data.shape[0]
    assert(type(k_fold) == int)
    assert(k_fold > 0 and k_fold <= num_row)

    np.random.shuffle(train_val_data)
    data_splits=np.vsplit(train_val_data,k_fold)

    for k in range(0, k_fold):
        # divide data to train/val set
        val_data = data_splits[k]
        if k==0:
            train_data = data_splits[0]
        else:
            train_data_splits=data_splits[0:k]+data_splits[k+1:k_fold]
            train_data = np.concatenate(train_data_splits)

        # train decision tree
        tree = Decision_tree()
        tree.train(train_data[:, :-1], train_data[:, -1])
        tree.prune(train_data[:, :-1], train_data[:, -1])

        yield tree,compute_confusion_matrix(tree, val_data, len(set(val_data[:, -1])))


def compute_confusion_matrix(tree, test_data, num_label):
    """
    compute confusion matrix of a learnt tree

    Args:
            tree	 (dict): learnt decision tree
            test_data (int): test data with label on the last column
            num_label (int): number of label category in dataset(row/col number of matrix)
    Outputs:
			confusion_matrix(np.ndarray,shape(num_label,num_label))
    """
    # assert(type(tree)==dict)
    assert(type(test_data) == np.ndarray)
    assert(len(test_data.shape) == 2 and test_data.shape[1] > 1)
    assert(type(num_label) == int)
    assert(num_label > 0)

    confusion_matrix = np.zeros([num_label, num_label])
    predicted_label = tree.classify(test_data[:, :-1])
    gt_label = test_data[:, -1]

    for p, g in zip(predicted_label, gt_label):
        confusion_matrix[int(p)-1, int(g)-1] += 1

    return confusion_matrix


def compute_precision(confusion_matrix):
	"""
    compute precision on each class

    Args:
            confusion_matrix (np.ndarry): square 2-D matrix
	Outputs:
	    precision_vec (np.array,lenghth=confusion_matrix's width)
	"""
	assert(type(confusion_matrix)==np.ndarray)
	assert(len(confusion_matrix.shape)==2)
	assert(confusion_matrix.shape[0]==confusion_matrix.shape[1])

	true_positive_vec=np.diag(confusion_matrix)
	return true_positive_vec/confusion_matrix.sum(axis=1)


def compute_recall(confusion_matrix):
	"""
    compute recall on each class

    Args:
            confusion_matrix (np.ndarry): square 2-D matrix

	Outputs:
	    f_measure (np.array,lenghth=confusion_matrix's width)
	"""
	assert(type(confusion_matrix)==np.ndarray)
	assert(len(confusion_matrix.shape)==2)
	assert(confusion_matrix.shape[0]==confusion_matrix.shape[1])

	true_positive_vec=np.diag(confusion_matrix)
	return confusion_matrix.sum(axis=0)


def compute_f_measure(confusion_matrix,beta=1.0):
	"""
    compute f_measure over a confusion matrix

    Args:
            confusion_matrix (np.ndarry): square 2-D matrix
	        beta (float): weight of precision
	Outputs:
	    f_measure (np.array,lenghth=confusion_matrix's width): f1_measure on each class
	"""
	assert(type(confusion_matrix)==np.ndarray)
	assert(len(confusion_matrix.shape)==2)
	assert(confusion_matrix.shape[0]==confusion_matrix.shape[1])
	assert(type(beta)==float or type(beta)==int)

	precision_vec=compute_precision(confusion_matrix)

	recall_vec=compute_recall(confusion_matrix)

	f_measure=(1+beta**2)*(precision_vec*recall_vec)/(beta**2*precision_vec+recall_vec)

	return f_measure


def main():
	#assert(len(sys.argv)==2)

	# load data from file and shuffle
	#train_val_data = np.loadtxt(sys.argv[1])
	train_val_data = np.loadtxt('wifi_db/clean_dataset.txt')
	np.random.shuffle(train_val_data)

	for i,(_,confusion_matrix) in enumerate(tree_cross_validate(train_val_data)):
		print(str(i)+'-th cross validation:')
		print('    precision:')
		print(compute_precision(confusion_matrix))
		print('    recall:')
		print(compute_recall(confusion_matrix))
		print('    f1 measure:')
		print(compute_f_measure(confusion_matrix))

	print('using all data to train a decision tree')
	tree = Decision_tree()
	tree.train(train_val_data[:, :-1], train_val_data[:, -1])
	tree.prune(train_val_data[:, :-1], train_val_data[:, -1])
	#tree.draw()


if __name__=="__main__":
	main()


